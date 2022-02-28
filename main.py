import os
from subprocess import getoutput

import torch
import yaml
from torchvision import datasets

from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from data.loader import TwoCropsTransform, GaussianBlur, Solarize
import torchvision.transforms as transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer
from argparse import ArgumentParser
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import builtins


# random_seed = 0
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# import numpy as np
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)
# import random
# random.seed(random_seed)

sys.path.append('../')

parser = ArgumentParser()
parser.add_argument('--datadir', type=str, default='/home/ykim/data/imagenet/')
parser.add_argument('--dataset', type=str, default='imagenet100')
parser.add_argument('--vessl', action='store_true')
parser.add_argument('--accum', type=int, default=1)

#Network args
parser.add_argument('--name', type=str, default="resnet18")
parser.add_argument('--hidden-dim', type=int, default=512)
parser.add_argument('--proj-size', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=1024)

#Trainer args
parser.add_argument('--max-epochs', type=int, default=200)
parser.add_argument('--warmup-epochs', type=int, default=10)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--m', type=float, default=0.996)

#Optimizer args
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--wd', type=float, default=1.5e-6)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--t', type=float, default=0.001, help="trust_coefficient")

parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--single', action='store_true')
parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
args = parser.parse_args()

class ImageNet100(datasets.ImageFolder):
    def __init__(self, root, split, transform):
        with open('./splits/imagenet100.txt') as f:
            classes = [line.strip() for line in f]
            class_to_idx = { cls: idx for idx, cls in enumerate(classes) }

        super().__init__(os.path.join(root, split), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

class ImageNet1000(datasets.ImageFolder):
    def __init__(self, root, split, transform):
        with open('../splits/imagenet1000.txt') as f:
            classes = [line.strip().split(':')[0] for line in f]
            class_to_idx = { cls: idx for idx, cls in enumerate(classes) }

        super().__init__(os.path.join(root, split), transform=transform)
        samples = []
        for path, label in self.samples:
            cls = self.classes[label]
            if cls not in class_to_idx:
                continue
            label = class_to_idx[cls]
            samples.append((path, label))

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]

class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])

def main_ddp(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    args.gpu = rank
    print(f"rank : {rank}, world_size : {world_size}")

    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    device = f'cuda:{args.gpu}'
    print(f"Training with: {device}")

    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    # online network
    online_network = ResNet18(args.name, args.hidden_dim, args.proj_size)

    # predictor network
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        mlp_hidden_size = args.hidden_dim, projection_size = args.proj_size)

    # target encoder
    target_network = ResNet18(args.name, args.hidden_dim, args.proj_size)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    if args.dataset == 'imagenet100':
        train_dataset =  ImageNet100(args.datadir, split='train', 
                                    transform=TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2)))
    elif args.dataset == 'imagenet1000':
        train_dataset =  ImageNet1000(args.datadir, split='train', 
                                transform=TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2)))

    online_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(online_network)
    predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(predictor)
    target_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(target_network)

    online_network.to(args.gpu)
    predictor.to(args.gpu)
    target_network.to(args.gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    
    online_network = torch.nn.parallel.DistributedDataParallel(online_network, device_ids=[args.gpu])
    predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[args.gpu])
    target_network = torch.nn.parallel.DistributedDataParallel(target_network, device_ids=[args.gpu])

    print("network initialization finished")
    
    args.lr = args.lr * args.batch_size / 256
    args.batch_size = int(args.batch_size // world_size)

    optimizer = LARS(list(online_network.parameters()) + list(predictor.parameters()), 
                        lr=args.lr, weight_decay=args.wd, momentum=args.momentum, trust_coefficient=args.t)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs - args.warmup_epochs)

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          args = args,
                          scheduler = scheduler)
    
    trainer.train(train_dataset)

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def main_single():

    args.gpu = 0
    device = f'cuda:{args.gpu}'
    print(f"Training with: {device}")

    # online network
    online_network = ResNet18(args.name, args.hidden_dim, args.proj_size)

    # predictor network
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                        mlp_hidden_size = args.hidden_dim, projection_size = args.proj_size)

    # target encoder
    target_network = ResNet18(args.name, args.hidden_dim, args.proj_size)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

    if args.dataset == 'imagenet100':
        train_dataset =  ImageNet100(args.datadir, split='train', 
                                    transform=TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2)))
    elif args.dataset == 'imagenet1000':
        train_dataset =  ImageNet1000(args.datadir, split='train', 
                                transform=TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2)))

    online_network = online_network.to_sequential().to(args.gpu)
    predictor = predictor.to_sequential().to(args.gpu)
    target_network = target_network.to_sequential().to(args.gpu)


    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    print("network initialization finished")
    
    #args.lr = args.lr * args.batch_size / 256
    optimizer = LARS(list(online_network.parameters()) + list(predictor.parameters()), 
                        lr=args.lr, weight_decay=args.wd, momentum=args.momentum, trust_coefficient=args.t)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs - args.warmup_epochs)
    scheduler = None

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          args = args,
                          scheduler = scheduler)
    
    trainer.train(train_dataset)

if __name__ == '__main__':
    if not args.single:
        run_demo(main_ddp, args.num_gpus)
    else:
        main_single()
    
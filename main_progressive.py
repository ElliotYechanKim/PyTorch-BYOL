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
from trainer import BYOLTrainer, ProgTrainer
from argparse import ArgumentParser
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import builtins
import torchvision

random_seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(random_seed)
torch.manual_seed(random_seed)
import random
random.seed(random_seed)

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
parser.add_argument('--optimizer', type=str, default='LARS')

parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--single', action='store_true')
parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
parser.add_argument('--progressive', action='store_true')

#Architecture args
parser.add_argument('--moco', action='store_true')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

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

def main_single():

    args.gpu = 0
    device = f'cuda:{args.gpu}'
    print(f"Training with: {device}")

    # online network
    online_network = ResNet18(args.name)
    online_network = online_network.to(args.gpu)
    
    if not args.moco:
        # predictor network
        predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features,
                            mlp_hidden_size = args.hidden_dim, projection_size = args.proj_size)
        predictor = predictor.to(args.gpu)
        optimizer_params = list(online_network.parameters()) + list(predictor.parameters())
    else:
        predictor = None
        optimizer_params = list(online_network.parameters())
    
    # target encoder
    target_network = ResNet18(args.name)
    target_network = target_network.to(args.gpu)

    mean = torch.tensor([0.43, 0.42, 0.39])
    std  = torch.tensor([0.27, 0.26, 0.27])
    normalize = transforms.Normalize(mean=mean, std=std)

    if not args.progressive:
        augmentation1 = [
            transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

        augmentation2 = [
            transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            # transforms.RandomApply([Solarize()], p=0.2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transforms_func = TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2))
    else:
        # augmentation1 = [
        #     transforms.ToTensor(),
        #     #normalize
        # ]
        # augmentation2 = [
        #     transforms.ToTensor(),
        #     #normalize
        # ]
        #transforms_func = TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2))
        transforms_func = TwoCropsTransform(np.array, np.array)

    if args.dataset == 'imagenet100':
        train_dataset =  ImageNet100(args.datadir, split='train', transform=transforms_func)
    elif args.dataset == 'imagenet1000':
        train_dataset =  torchvision.datasets.ImageNet(args.datadir, split='train', transform=transforms_func)
    elif args.dataset == 'stl10':
        train_dataset =  torchvision.datasets.STL10(args.datadir, split='train+unlabeled', transform=transforms_func)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    print("network initialization finished")
    
    #args.lr = args.lr * args.batch_size / 256
    if args.optimizer == 'LARS':
        optimizer = LARS(optimizer_params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum, trust_coefficient=args.t)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs - args.warmup_epochs)
    
    if args.progressive:
        trainer = ProgTrainer(online_network=online_network,
                            target_network=target_network,
                            optimizer=optimizer,
                            predictor=predictor,
                            args = args,
                            scheduler = scheduler)
    else:
        trainer = BYOLTrainer(online_network=online_network,
                            target_network=target_network,
                            optimizer=optimizer,
                            predictor=predictor,
                            args = args,
                            scheduler = scheduler)
    
    trainer.train(train_dataset)

if __name__ == '__main__':
    main_single()
    
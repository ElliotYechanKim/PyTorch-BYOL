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

sys.path.append('../')
torch.manual_seed(0)

parser = ArgumentParser()
parser.add_argument('--datadir', type=str, default='/home/ykim/data/imagenet/')
parser.add_argument('--vessl', action='store_true')
parser.add_argument('--accum', type=int, default=1)
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

def main_ddp(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    args.gpu = rank
    print(f"rank : {rank}, world_size : {world_size}")

    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = f'cuda:{args.gpu}'
    print(f"Training with: {device}")

    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

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

    # data_transform = get_simclr_data_transforms(**config['data_transforms'])
    # train_dataset = datasets.STL10('/home/thalles/Downloads/', split='train+unlabeled', download=True,
    #                                transform=MultiViewDataInjector([data_transform, data_transform]))

    train_dataset =  ImageNet100(args.datadir, split='train', 
                                transform=TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2)))
    
    # online network
    online_network = ResNet18(**config['network'])
    
    # pretrained_folder = config['network']['fine_tune_from']

    # # load pre-trained model if defined
    # if pretrained_folder:
    #     try:
    #         checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

    #         # load pre-trained parameters
    #         load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
    #                                  map_location=torch.device(torch.device(device)))

    #         online_network.load_state_dict(load_params['online_network_state_dict'])

    #     except FileNotFoundError:
    #         print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head'])

    # target encoder
    target_network = ResNet18(**config['network'])

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

    config['optimizer']['params']['lr'] = config['optimizer']['params']['lr'] * config['trainer']['batch_size'] / 256
    args.lr = config['optimizer']['params']['lr']

    config['trainer']['batch_size'] = int(config['trainer']['batch_size'] // world_size)

    optimizer = LARS(list(online_network.parameters()) + list(predictor.parameters()), **config['optimizer']['params'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['trainer']['max_epochs'] - config['trainer']['warmup_epochs'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          args = args,
                          scheduler = scheduler,
                          **config['trainer'])
    
    trainer.train(train_dataset)

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    run_demo(main_ddp, torch.cuda.device_count())
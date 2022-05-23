import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import torchvision.transforms as transforms
import numpy as np
import random

from torchvision import datasets
from torch.utils.data import Dataset
from data.imagenet100 import ImageNet100
from data.loader import TwoCropsTransform, GaussianBlur, Solarize
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import Trainer
from argparse import ArgumentParser
from lars import LARS


sys.path.append('../')

parser = ArgumentParser()
parser.add_argument('--datadir', type=str, default='/home/ykim/data/imagenet/')
parser.add_argument('--dataset', type=str, default='imagenet100')
parser.add_argument('--vessl', action='store_true')

#Network args
parser.add_argument('--name', type=str, default="resnet18")
parser.add_argument('--hidden-dim', type=int, default=512)
parser.add_argument('--proj-size', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=1024)

#Optimizer args
parser.add_argument('--optimizer', type=str, default='LARS')
parser.add_argument('--lr', type=float, default=0.2)
parser.add_argument('--wd', type=float, default=1.5e-6)
parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
parser.add_argument('--t', type=float, default=0.001, help="trust_coefficient")
parser.add_argument('--accum', type=int, default=1, help='Number of accumulation steps')
parser.add_argument('--m', type=float, default=0.996, help = 'target momenutm')

#Architecture args
parser.add_argument('--moco', action='store_true')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

#Trainer args
parser.add_argument('--max-epochs', type=int, default=200)
parser.add_argument('--warmup-epochs', type=int, default=10)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--single', action='store_true')
parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())

#Progressive args
parser.add_argument('--fix-random', action='store_true', help = 'Fix the seeds for debugging')
parser.add_argument('--progressive', action='store_true')
parser.add_argument('--init-prob',  type=float, default = 0.5)
parser.add_argument('--max-prob',  type=float, default = 1.25)
parser.add_argument('--filter-ratio',  type=float, default = 0.1)
parser.add_argument('--interpolate', type=str, default = 'linear')
parser.add_argument('--sim-pretrained', action='store_true', help = 'Using pre-trained model to masuer the similarity')

args = parser.parse_args()

if args.fix_random:
    random_seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

class ProgressiveDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.increase_ratio(0) # initial ratio

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def increase_ratio(self, epoch):

        if args.interpolate == 'linear':
            s = args.init_prob + (args.max_prob - args.init_prob) / args.max_epochs * epoch
        elif args.interpolate == 'log':
            s = np.exp(np.log(args.init_prob) + (np.log(args.max_prob - args.init_prob) - np.log(args.init_prob)) / \
                                                    np.log(args.max_epochs) * np.log(epoch)) + args.init_prob
        scale_lower = (1 - max(s, 1)) + 0.08
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        normalize = transforms.Normalize(mean=mean, std=std)

        augmentation1 = [
            transforms.RandomResizedCrop(args.orig_img_size * s, scale=(scale_lower, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1 * s)  # not strengthened
            ], p=0.8 * s),
            transforms.RandomGrayscale(p = 0.2 * s),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p = 1.0 * s),
            transforms.RandomHorizontalFlip(p = 0.5 * s),
            transforms.ToTensor(),
            normalize
        ]

        augmentation2 = [
            transforms.RandomResizedCrop(args.orig_img_size * s, scale=(scale_lower, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1 * s)  # not strengthened
            ], p=0.8 * s),
            transforms.RandomGrayscale(p = 0.2 * s),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p = 0.1 * s),
            transforms.RandomApply([Solarize()], p = 0.2 * s),
            transforms.RandomHorizontalFlip(p = 0.5 * s),
            transforms.ToTensor(),
            normalize
        ]
        transforms_func = TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2))
        self.dataset.transform = transforms_func

def main_single():

    args.gpu = 0
    device = f'cuda:{args.gpu}'
    print(f"Training with: {device}")

    # online network
    online_network = ResNet18(args.name)
    online_network = online_network.to(args.gpu)
    
    if not args.moco:
        # predictor network
        predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features, name=args.name)
        predictor = predictor.to(args.gpu)
        optimizer_params = list(online_network.parameters()) + list(predictor.parameters())
    else:
        predictor = None
        optimizer_params = list(online_network.parameters())
    
    # target encoder
    target_network = ResNet18(args.name)
    target_network = target_network.to(args.gpu)

    if args.dataset == 'imagenet100':
        train_dataset =  ImageNet100(args.datadir, split='train')
        args.orig_img_size = 224
    elif args.dataset == 'imagenet1000':
        train_dataset =  datasets.ImageNet(args.datadir, split='train')
        args.orig_img_size = 224
    elif args.dataset == 'stl10':
        train_dataset =  datasets.STL10(args.datadir, split='train+unlabeled')
        args.orig_img_size = 96

    if args.progressive:
        train_dataset = ProgressiveDataset(train_dataset)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    print("network initialization finished")
    
    #Lineary Scalining the learning rate
    args.lr = args.lr * args.batch_size / 256
    
    #Accumulate batches
    args.batch_size = int(args.batch_size / args.accum)
    print(f"Adjusted batch size after accum: {args.batch_size}")

    if args.optimizer == 'LARS':
        optimizer = LARS(optimizer_params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum, trust_coefficient=args.t)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)
    
    trainer = Trainer(online_network=online_network,
                        target_network=target_network,
                        optimizer=optimizer,
                        predictor=predictor,
                        args = args)
    
    trainer.train(train_dataset)

if __name__ == '__main__':
    main_single()
    
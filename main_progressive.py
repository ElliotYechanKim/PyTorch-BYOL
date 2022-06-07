import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import sys
import numpy as np
import random
import builtins
import wandb
import math

from torchvision import datasets
from data.imagenet100 import ImageNet100
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import Trainer
from argparse import ArgumentParser
from lars import LARS
from data.dataset import ProgressiveDataset
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../')

parser = ArgumentParser()
parser.add_argument('--datadir', type=str, default='/home/ykim/data/stl10')
parser.add_argument('--dataset', type=str, default='stl10')
parser.add_argument('--vessl', action='store_true')

#Network args
parser.add_argument('--name', type=str, default="resnet18")
parser.add_argument('--hidden-dim', type=int, default=512)
parser.add_argument('--proj-size', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=512)

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
parser.add_argument('--max-epochs', type=int, default=40)
parser.add_argument('--warmup-epochs', type=int, default=0)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")

#DDP args
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
parser.add_argument('--filter-type', type=str, default = 'window')
args = parser.parse_args()

if args.fix_random:
    random_seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

def main_single():
    args.gpu = 0
    device = f'cuda:{args.gpu}'
    print(f"Training with: {device}")
    args.wandb = wandb.init(project="progressive")

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
        args.orig_img_size = 64

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
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(optimizer_params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
        
    if args.progressive:
        args.sigma3 = math.ceil(args.batch_size * 0.03)
        args.orig_batch_size = args.batch_size
        args.batch_size = int(args.batch_size / (1 - args.filter_ratio)) + 2 * args.sigma3
        
        #DROP LAST
        orig_updates = (len(train_dataset) // args.orig_batch_size) * args.max_epochs
        args.total_iter = orig_updates
        our_updates = (len(train_dataset) // args.batch_size) * args.max_epochs
        added_epochs = (orig_updates - our_updates) / (len(train_dataset) // args.batch_size)
        args.max_epochs += int(math.ceil(added_epochs))
    else:
        args.init_prob = 1

    args.wandb.config.update(args)
    args.wandb.watch([online_network, target_network, predictor])
    print(args.wandb.config)
    
    train_dataset = ProgressiveDataset(train_dataset, args.wandb.config)
    
    writer = SummaryWriter() if args.gpu == 0 else None
    train_dataset.increase_stage(0, writer, args.wandb)
    
    trainer = Trainer(online_network=online_network,
                        target_network=target_network,
                        optimizer=optimizer,
                        predictor=predictor,
                        args = args.wandb.config,
                        wb = args.wandb,
                        writer = writer)
    
    trainer.train(train_dataset)

def main_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

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
    online_network = ResNet18(args.name)
    online_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(online_network)
    online_network = online_network.to(args.gpu)
    online_network = torch.nn.parallel.DistributedDataParallel(online_network, device_ids=[args.gpu])
    
    if not args.moco:
        # predictor network
        predictor = MLPHead(in_channels=online_network.module.projection.net[-1].out_features, name=args.name)
        predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(predictor)
        predictor = predictor.to(args.gpu)
        predictor = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[args.gpu])
        optimizer_params = list(online_network.parameters()) + list(predictor.parameters())
    else:
        predictor = None
        optimizer_params = list(online_network.parameters())
    
    # target encoder
    target_network = ResNet18(args.name)
    target_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(online_network)
    target_network = target_network.to(args.gpu)
    target_network = torch.nn.parallel.DistributedDataParallel(target_network, device_ids=[args.gpu])

    print("network initialization finished")
    
    if args.dataset == 'imagenet100':
        train_dataset =  ImageNet100(args.datadir, split='train')
        args.orig_img_size = 224
    elif args.dataset == 'imagenet1000':
        train_dataset =  datasets.ImageNet(args.datadir, split='train')
        args.orig_img_size = 224
    elif args.dataset == 'stl10':
        train_dataset =  datasets.STL10(args.datadir, split='train+unlabeled')
        args.orig_img_size = 96
    
    #Lineary Scalining the learning rate
    args.lr = args.lr * args.batch_size / 256
    
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size // world_size)

    # DDP, need to adjust workers
    args.num_workers = int((args.num_workers + world_size - 1) / world_size)

    #Accumulate batches
    args.batch_size = int(args.batch_size / args.accum)
    print(f"Adjusted batch size after DDP & accum: {args.batch_size}")

    if args.optimizer == 'LARS':
        optimizer = LARS(optimizer_params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum, trust_coefficient=args.t)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)
    
    if args.progressive:
        args.sigma3 = math.ceil(args.batch_size * 0.03)
        args.orig_batch_size = args.batch_size
        args.batch_size = int(args.batch_size / (1 - args.filter_ratio)) + 2 * args.sigma3
        
        orig_updates = len(train_dataset) / args.orig_batch_size
        updates = len(train_dataset) // args.batch_size
        added_epochs = (orig_updates - updates) * args.max_epochs / updates
        args.extra_iter = math.ceil(added_epochs) - added_epochs
        args.max_epochs += math.ceil(added_epochs)
    
    if args.gpu == 0:
        args.wandb = wandb.init(config=args, project="progressive")
        args.wandb.watch([online_network, target_network, predictor])
        print(args.wandb.config)
    
    writer = SummaryWriter() if args.gpu == 0 else None
    
    train_dataset = ProgressiveDataset(train_dataset, args.wandb.config)

    trainer = Trainer(online_network=online_network,
                        target_network=target_network,
                        optimizer=optimizer,
                        predictor=predictor,
                        args = args,
                        wb = args.wandb,
                        writer = writer)
    print('trainer initialization finished')
    
    trainer.train(train_dataset)


def run_ddp(main_fn, world_size):
    mp.spawn(main_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    if args.single:
        main_single()
    else:
        run_ddp(main_ddp, args.num_gpus)
    
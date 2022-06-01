import torch
import sys
from torchvision import transforms
import torchvision
import yaml
import os
sys.path.append('../')
from models.resnet_base_network import ResNet18
from argparse import ArgumentParser
from torchvision.datasets import STL10, CIFAR10, CIFAR100, ImageFolder
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict, OrderedDict
import time
import math
from defer_batchnorm import DeferredBatchNorm
import torch.multiprocessing as mp
import torch.distributed as dist
import builtins
from torch.utils.tensorboard import SummaryWriter
from data.imagenet100 import ImageNet100

parser = ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--rundir', help='path to model')
parser.add_argument('--datadir', default = "/home/ykim/data/imagenet/", metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default = "imagenet100")
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--warmup-epochs', default=0, type=int, metavar='N',
                    help='number of warmup epochs to run')
parser.add_argument('--optimizer', default='SGD', type=str)        
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.2, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--single', action="store_true")
parser.add_argument('--model-num', type=int, default=0)
parser.add_argument('--knn', action='store_true')
parser.add_argument('--single-knn', action="store_true")
parser.add_argument('--finetune', action="store_true")

parser.add_argument('--network', default='resnet18', type=str)
parser.add_argument('--train-epochs', default=200, type=int)
best_acc1 = 0
best_acc5 = 0

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

def load_model(args, config) -> torch.nn.Sequential:
    
    encoder = ResNet18(config['network']['name'])
    output_feature_dim = encoder.projection.net[0].in_features

    if not args.single:
        models = defaultdict(list)
        for root, subdirs, files in os.walk(os.path.join(args.rundir, 'checkpoints')):
            for fn in files:
                if fn.startswith(f'model_batch{args.model_num}_') and '_part' in fn and fn.endswith('.pt'):
                    model_prefix = fn.split('_part')[0]
                    models[model_prefix].append(os.path.join(root, fn))

        available_model_prefix_lst = list(sorted(models.keys()))
        print(available_model_prefix_lst)
        
        param = OrderedDict()
        
        # careful not to load projection
        max_layer_num = 0
        for model_path in sorted(models[available_model_prefix_lst[0]]):
            #print(model_path)
            load_params = torch.load(model_path, map_location=torch.device('cuda'))                    
            if 'target_network_state_dict' in load_params:
                new_params: OrderedDict = load_params['target_network_state_dict']
                for layer_name in new_params.keys():
                    #print(layer_name)
                    layer_num = int(layer_name.split('.')[0])
                    if layer_num >= max_layer_num:
                        param[layer_name] = new_params[layer_name]
                        max_layer_num = max(max_layer_num, layer_num)

        encoder = encoder.to_sequential()
        DeferredBatchNorm.convert_deferred_batch_norm(encoder, 1)
        encoder.load_state_dict(param)
        #print("Parameters successfully loaded from target_network_state_dict.")
        DeferredBatchNorm.convert_batch_norm_deffered(encoder)
        encoder = torch.nn.Sequential(*list(encoder.children())[:-5])
    else:
        param = OrderedDict()
        load_params = torch.load(os.path.join(args.rundir, f'checkpoints/model{args.model_num}.pth'),
                            map_location=torch.device('cuda'))
        #max_layer_num = 0
        new_params: OrderedDict = load_params['target_network_state_dict']
        # for layer_name in new_params.keys():
        #     # print(layer_name)
        #     layer_name_split = layer_name.split('.')[1:]
        #     if layer_name_split[0] == "projetion":
        #         layer_name_split[0] = "projection"
        #     new_layer_name = '.'.join(e for e in layer_name_split)
        #     param[new_layer_name] = new_params[layer_name]
        # encoder.load_state_dict(param)
        #print("Parameters successfully loaded from target_network_state_dict.")
        # remove the projection head
        encoder.load_state_dict(new_params)
        encoder = torch.nn.Sequential(*list(encoder)[:-5])
    
    print(encoder)
    logreg = LogisticRegression(output_feature_dim, args.num_classes)
    return encoder, logreg

def run_main(main_fn, world_size):
    mp.spawn(main_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def main_ddp(rank, world_size):
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    global best_acc1
    global best_acc5
    
    args = parser.parse_args()

    args.gpu = rank
    print(f"rank : {rank}, world_size : {world_size}")

    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if not args.single:
        config = yaml.load(open(os.path.join(args.rundir, 'checkpoints/config.yaml'), "r"), Loader=yaml.FullLoader)
    else:
        config = dict()
        config['network'] = dict()
        config['network']['projection_head'] = dict()
        config['trainer'] = dict()
        if args.network == 'resnet18':
            config['network']['name'] = args.network
            config['network']['projection_head']['mlp_hidden_size'] = 512
            config['network']['projection_head']['projection_size'] = 128
        elif args.network == 'resnet50':
            config['network']['name'] = args.network
            config['network']['projection_head']['mlp_hidden_size'] = 2048
            config['network']['projection_head']['projection_size'] = 256
        config['trainer']['max_epochs'] = args.train_epochs
    
    # Finetune learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256
    args.batch_size = int(args.batch_size // world_size)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    if args.dataset == "imagenet100":
        train_dataset = ImageNet100(args.datadir, split='train', transform=train_transform)
        val_dataset = ImageNet100(args.datadir, split='val', transform=test_transform)
        args.num_classes = 100
    elif args.dataset == 'stl10':
        train_dataset = STL10(root=args.datadir, split='train', transform=train_transform, download = True)
        val_dataset = STL10(root=args.datadir, split='test', transform=test_transform, download = True)
        args.num_classes = 10
    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(root=args.datadir, train=True, transform=train_transform, download = True)
        val_dataset = CIFAR10(root=args.datadir, train=False, transform=test_transform, download = True)
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = CIFAR100(root=args.datadir, train=True, transform=train_transform, download = True)
        val_dataset = CIFAR100(root=args.datadir, train=False, transform=test_transform, download = True)
        args.num_classes = 100
    elif args.dataset == 'imagenet1000':
        train_dataset = torchvision.datasets.ImageNet(args.datadir, split='train', transform=train_transform)
        val_dataset = torchvision.datasets.ImageNet(args.datadir, split='val', transform=test_transform)
        args.num_classes = 1000

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), drop_last = False,
        sampler=train_sampler, num_workers=args.workers)
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None), drop_last = False,
        sampler=val_sampler, num_workers=args.workers)

    encoder, logreg = load_model(args, config)

    encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder).to(args.gpu)
    logreg = logreg.to(args.gpu)
    encoder = torch.nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu])
    logreg = torch.nn.parallel.DistributedDataParallel(logreg, device_ids=[args.gpu])

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(args.gpu)

    if args.finetune:
        parameters = list(encoder.parameters()) + list(logreg.parameters())
    else:
        parameters = logreg.parameters()
    
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(parameters, init_lr, momentum = 0.9,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(parameters, init_lr,
                                    weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, encoder, logreg, criterion, args)
        return
    
    writer = SummaryWriter() if args.gpu == 0 else None
    best_acc1_ls = []
    best_acc5_ls = []
    for epoch in range(args.epochs):
        
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, init_lr, epoch, writer, args)

        # train for one epoch
        train(train_loader, encoder, logreg, criterion, optimizer, epoch, writer, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, encoder, logreg, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)

        if args.gpu == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'encoder_state_dict': encoder.state_dict(),
                'logreg_state_dict' : logreg.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args)
        print(f"Best top 1 Accuracy on epoch {epoch} : {best_acc1}")
        print(f"Best top 5 Accuracy on epoch {epoch} : {best_acc5}")

        if epoch % 10 == 0:
            best_acc1_ls.append(best_acc1)
            best_acc5_ls.append(best_acc5)
    
    best_acc1_ls.append(best_acc1)
    best_acc5_ls.append(best_acc5)

    print(f"Best top 1 Accuarcay List : {best_acc1_ls}")
    print(f"Best top 5 Accuarcay List : {best_acc5_ls}")


def train(train_loader, encoder, logreg, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """

    # optimize Both for Finetuning
    if args.finetune:
        encoder.train()
        logreg.train()
    else:
        encoder.eval()
        logreg.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(args.gpu)
        target = target.to(args.gpu)

        # compute output
        if args.finetune:
            feature_vector_x = encoder(images)
            feature_vector_x = torch.squeeze(feature_vector_x)
        else:
            with torch.no_grad():
                feature_vector_x = encoder(images)
                feature_vector_x = torch.squeeze(feature_vector_x)
        output = logreg(feature_vector_x)
        loss = criterion(output, target)
        
        if args.gpu == 0:
            writer.add_scalar('loss', loss.item(), global_step=(epoch * len(train_loader) + i))
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, encoder, logreg, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    encoder.eval()
    logreg.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(args.gpu)
            target = target.to(args.gpu)

            # compute output
            with torch.no_grad():
                feature_vector_x = encoder(images)
                feature_vector_x = torch.squeeze(feature_vector_x)
                output = logreg(feature_vector_x)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def save_checkpoint(state, is_best, args):
    if is_best:
        filename =  os.path.join(args.rundir, f'checkpoints/eval_model.pth')
        torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, writer, args):
    """Decay the learning rate based on schedule"""
    if epoch < args.warmup_epochs:
        cur_lr = init_lr * (epoch + 1) / args.warmup_epochs
    else:
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    if args.gpu == 0:
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    run_main(main_ddp, torch.cuda.device_count())
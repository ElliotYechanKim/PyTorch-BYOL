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
from torch.utils.data import random_split
from PIL import Image
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict, OrderedDict
from typing import List
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import torch.nn as nn
import time
import math
from tqdm import tqdm
import shutil

parser = ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--rundir', help='path to model')
parser.add_argument('--datadir', default = "/home/ykim/data/imagenet/", metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', default = "imagenet100")
# parser.add_argument('-c', '--num-classes', default=100, type=int, metavar='N',
#                     help='number of classes to classify')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
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
parser.add_argument('--network', default='resnet18', type=str)
parser.add_argument('--train-epochs', default=200, type=int)

best_acc1 = 0
best_acc5 = 0

class ImageNet100(ImageFolder):
    def __init__(self, root, split, transform):
        with open('../splits/imagenet100.txt') as f:
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

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)

def load_model(args) -> torch.nn.Sequential:
    
    encoder = ResNet18(args.network)
    output_feature_dim = encoder.projection.net[0].in_features

    load_params = torch.load(os.path.join(args.rundir, f'checkpoints/model{args.model_num}.pth'),
                        map_location=torch.device('cuda'))
    
    new_params: OrderedDict = load_params['target_network_state_dict']
    encoder.load_state_dict(new_params)
    
    print("Parameters successfully loaded from target_network_state_dict.")
    # remove the projection head
    encoder = torch.nn.Sequential(*list(encoder)[:-5]).to('cuda')
    logreg = LogisticRegression(output_feature_dim, args.num_classes).to('cuda')

    return encoder, logreg


def main():
    global best_acc1
    global best_acc5
    
    args = parser.parse_args()

    # Data loading code
    mean = torch.tensor([0.43, 0.42, 0.39])
    std  = torch.tensor([0.27, 0.26, 0.27])
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    test_transform = transforms.Compose([
            transforms.CenterCrop(96),
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.single_knn:
        single_evaluate(train_loader, val_loader, args)
        return

    if args.knn:
        evaulte_knn(train_loader, val_loader, args)
        return
    
    encoder, logreg = load_model(args)
    
    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimize only the linear classifier
    encoder.eval()
    logreg.train()

    optimizer = torch.optim.SGD(logreg.parameters(), init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(val_loader, encoder, logreg, criterion, args)
        return

    best_acc1_ls = []
    best_acc5_ls = []
    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, encoder, logreg, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, encoder, logreg, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)

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


def train(train_loader, encoder, logreg, criterion, optimizer, epoch, args):
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
    encoder.eval()
    logreg.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

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
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

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

def nn_evaluator(val_loader, test_loader, encoder):
    
    with torch.no_grad():
        features = []
        labels   = []
        for i, (x, y) in enumerate(val_loader):
            z = encoder(x.cuda())
            z = z.squeeze()
            z = F.normalize(z, dim=-1)
    
            features.append(z.detach())
            labels.append(y.detach())
        
        features = torch.cat(features, 0).detach()
        labels = torch.cat(labels, 0).detach()

        corrects, total = 0, 0
        for x, y in test_loader:
            z = F.normalize(encoder(x.cuda()).squeeze(), dim=-1)
            scores = torch.einsum('ik, jk -> ij', z, features)
            preds = labels[scores.argmax(1)]
            corrects += (preds.cpu() == y).long().sum().item()
            total += y.shape[0]
    result = 100 * (corrects / total)
    return result

def evaulte_knn(val_loader, test_loader, args, config):
    max_epochs = config['trainer']['max_epochs']
    
    initial_model_num = args.model_num

    specific_model_num_list = [400,]
    accuracy_list = []
    for i in range(max_epochs):
        if i % 10 == 0 or i == max_epochs - 1:
            if args.single:
                args.model_num = i
            else:
                args.model_num = initial_model_num * (i + 1)
            encoder, logreg = load_model(args, config)
            print(f"model{i} loaded")
            accuracy = nn_evaluator(val_loader, test_loader, encoder)
            print(f"{i} epoch accuracy : {accuracy}")
            accuracy_list.append(accuracy)
    
    print(accuracy_list)

def single_evaluate(val_loader, test_loader, args, config):
    encoder, logreg = load_model(args, config)
    print(f"model{args.model_num} loaded")
    accuracy = nn_evaluator(val_loader, test_loader, encoder)
    print(f"Accuracy : {accuracy}")

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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
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
    main()
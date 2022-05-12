import os
import torch
import torchvision
import time
import math
import torch.nn.functional as F
import kornia.augmentation as K
import torchvision.transforms as transforms

from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from kornia.augmentation.container import AugmentationSequential
from data.loader import TwoCropsTransform, GaussianBlur, Solarize
from utils import _create_model_training_folder
from tqdm import tqdm

class Trainer:
    def __init__(self, online_network, target_network, predictor, optimizer, args):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.predictor = predictor
        
        if args.vessl:
            import vessl
            vessl.init(tensorboard = True)
        
        self.writer = SummaryWriter() if args.gpu == 0 else None
        self.initial_m = args.m
        self.args = args
        
        if self.writer:
            _create_model_training_folder(self.writer, files_to_same=["main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.args.m + param_q.data * (1. - self.args.m)

    def regression_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def contrastive_loss(self, online_view_1, online_view_2, target_view_1, target_view_2):
        def loss(q, k):
            q = torch.nn.functional.normalize(q, dim=1)
            k = torch.nn.functional.normalize(k, dim=1)
            logits = torch.einsum('nc,mc->nm', [q, k]) / self.args.moco_t
            dev = torch.get_device(logits)
            N = logits.shape[0]  # batch size per GPU
            labels = torch.arange(N, dtype=torch.long).to(dev) # (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
            return torch.nn.CrossEntropyLoss()(logits, labels) * (2 * self.args.moco_t)

        return loss(online_view_1, target_view_2) + loss(online_view_2, target_view_1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
        
        if not self.args.single:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        self.batch_size = int(self.args.batch_size / self.args.accum)
        print(f"Adjusted batch size : {self.batch_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.args.num_workers, drop_last=False, shuffle=(train_sampler is None),
                                  sampler = train_sampler)

        niter = 0
        if self.args.gpu == 0:
            model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()


        # if self.args.progressive:
        #     adjust_period = self.args.max_epochs // self.args.num_stages
        #     print(adjust_period)
        
        for epoch_counter in range(self.args.max_epochs):
            if not self.args.single:
                train_sampler.set_epoch(epoch_counter)

            lr = self.adjust_learning_rate(epoch_counter + 1)
            m = self.adjust_momentum(epoch_counter)
            if self.args.gpu == 0:
                self.writer.add_scalar('learning_rate', lr, global_step=epoch_counter)
                self.writer.add_scalar('momentum', m, global_step=epoch_counter)
            
            niter = self.train_single(train_loader, niter, epoch_counter)

            if self.args.progressive:
                train_dataset.increase_stage(epoch_counter + 1)
            
            # save checkpoints
            if self.args.gpu == 0:
                self.save_model(os.path.join(model_checkpoints_folder, f'model{epoch_counter}.pth'))
        
        if self.args.gpu == 0:
            self.writer.close()

    def train_single(self, train_loader, niter, epoch):
        
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        print_list = [batch_time, data_time, losses]
        
        progress = ProgressMeter(
            len(train_loader),
            print_list,
            prefix="Epoch: [{}]".format(epoch)
        )

        end = time.time()
        for i, ((batch_view_1, batch_view_2), _) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            
            batch_view_1 = batch_view_1.to(self.args.gpu)
            batch_view_2 = batch_view_2.to(self.args.gpu)

            # if self.args.progressive:
            #     batch_view_1, batch_view_2 = self.adjust_augment_ratio(batch_view_1, batch_view_2, 
            #                                         niter, len(train_loader), self.args.max_epochs)
            #     aug_time.update(time.time() - end)
                
            # if self.args.gpu == 0 and i == 0:
            #     grid = torchvision.utils.make_grid(batch_view_1[:32])
            #     self.writer.add_image('views_1', grid, global_step=epoch)

            #     grid = torchvision.utils.make_grid(batch_view_2[:32])
            #     self.writer.add_image('views_2', grid, global_step=epoch)
        
            loss = self.update(batch_view_1, batch_view_2)
            loss = loss / self.args.accum

            loss.backward()

            if (i + 1) % self.args.accum == 0 or ((i + 1) == len(train_loader)): # Wait for several backward steps
                if self.args.gpu == 0:
                    self.writer.add_scalar('loss', (loss.item() * self.args.accum), global_step=niter)
                
                losses.update(loss.item(), batch_view_1.size(0))

                self.optimizer.step()
                self.optimizer.zero_grad()
                self._update_target_network_parameters() # update the key encoder
                
                # measure elapsed time
                batch_time.update(time.time() - end)
    

                if self.args.gpu == 0 and i % self.args.print_freq == 0:
                    progress.display(i)

                niter += 1
                end = time.time()
        
        return niter
    
    def update(self, batch_view_1, batch_view_2):
        if self.args.moco:
            predictions_from_view_1 = self.online_network(batch_view_1)
            predictions_from_view_2 = self.online_network(batch_view_2)
        else:
            predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
            predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
        
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        if self.args.moco:
            loss = self.contrastive_loss(predictions_from_view_1, predictions_from_view_2, targets_to_view_2, targets_to_view_1)
        else:
            loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
            loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
            loss = loss.mean()
        return loss

    def adjust_learning_rate(self, epoch):
        """Decays the learning rate with half-cycle cosine after warmup"""
        if epoch < self.args.warmup_epochs:
            lr = self.args.lr * epoch / self.args.warmup_epochs
        else:
            lr = self.args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - self.args.warmup_epochs) / (self.args.max_epochs - self.args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def adjust_momentum(self, epoch):
        self.m = 1. - (1. - self.initial_m) * (1. + math.cos(math.pi * epoch / self.args.max_epochs)) * 0.5
        return self.m
    
    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

    def adjust_augment_ratio(self, batch_view_1, batch_view_2, niter, length, epochs):
        
        init_size = self.args.orig_size * self.args.init_prob
        image_size = int(init_size + (self.args.orig_size - init_size) * niter / (length * epochs))
        #image_size = 96
        
        s = self.args.init_prob + (1 - self.args.init_prob) * niter / (length * epochs)
        s = 1

        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])

        augmentation1 = AugmentationSequential(
            K.RandomResizedCrop((image_size, image_size), scale=(0.08, 1.)),
            K.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1* s, p = 0.8 * s),
            K.RandomGrayscale(p = 0.2 * s),
            K.RandomGaussianBlur((9, 9), (0.1, 2.0), p = 1.0 * s), # 9 for STL-10 / 23 for ImageNet
            K.RandomHorizontalFlip(p = 0.5 * s),
            K.Normalize(mean=mean, std=std)
        )
        augmentation2 = AugmentationSequential(
            K.RandomResizedCrop((image_size, image_size), scale=(0.08, 1.)),
            K.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1* s, p = 0.8 * s),
            K.RandomGrayscale(p = 0.2 * s),
            K.RandomGaussianBlur((9, 9), (0.1, 2.0), p = 0.1 * s),
            K.RandomSolarize(p = 0.2 * s),
            K.RandomHorizontalFlip(p = 0.5 * s),
            K.Normalize(mean=mean, std=std)
        )
        return augmentation1(batch_view_1), augmentation2(batch_view_2)

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

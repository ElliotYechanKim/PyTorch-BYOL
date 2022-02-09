import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import _create_model_training_folder
import math
from tqdm import tqdm

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, args, scheduler):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.predictor = predictor
        self.max_epochs = args.max_epochs
        
        if args.vessl:
            import vessl
            vessl.init(tensorboard=True)
        self.writer = SummaryWriter() if args.gpu == 0 else None
        self.m = args.m
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.gpu = args.gpu
        self.warmup_epochs = args.warmup_epochs
        self.scheduler = scheduler
        self.lr = args.lr
        self.accumulation_steps = args.accum
        self.print_freq = args.print_freq
        if self.writer:
            _create_model_training_folder(self.writer, files_to_same=["main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self, epoch):
        """
        Momentum update of the key encoder
        """
        new_m = 1. - (1. - self.m) * (1. + math.cos(math.pi * epoch / self.max_epochs)) * 0.5
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * new_m + param_q.data * (1. - new_m)

    def regression_loss(self, x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        
        self.batch_size = int(self.batch_size / self.accumulation_steps)
        print(f"Adjusted batch size : {self.batch_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=False, 
                                  sampler = train_sampler)

       #pbar = tqdm(train_loader)
        niter = 0
        if self.gpu == 0:
            model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):

            train_sampler.set_epoch(epoch_counter)

            for i, ((batch_view_1, batch_view_2), _) in enumerate(train_loader):
                batch_view_1 = batch_view_1.to(self.gpu)
                batch_view_2 = batch_view_2.to(self.gpu)

                # if niter == 0 and self.gpu == 0:
                #     grid = torchvision.utils.make_grid(batch_view_1[:32])
                #     self.writer.add_image('views_1', grid, global_step=niter)

                #     grid = torchvision.utils.make_grid(batch_view_2[:32])
                #     self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2)
                loss = loss / self.accumulation_steps
                loss.backward()
                #pbar.set_postfix({'loss' : loss.item()})
                if (i + 1) % self.accumulation_steps == 0 or ((i + 1) == len(train_loader)):             # Wait for several backward steps
                    if self.gpu == 0:
                        self.writer.add_scalar('loss', (loss.item() * self.accumulation_steps), global_step=niter)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_target_network_parameters(epoch_counter)  # update the key encoder
                    if self.gpu == 0 and niter % self.print_freq == 0:
                        print("Loss on {}: {}".format(niter, (loss.item() * self.accumulation_steps)))
                    niter += 1
            print("End of epoch {}, loss : {}".format(epoch_counter, (loss.item() * self.accumulation_steps)))
            
            self.adjust_learning_rate(epoch_counter)
            
            # save checkpoints
            if self.gpu == 0:
                self.save_model(os.path.join(model_checkpoints_folder, f'model{epoch_counter}.pth'))
        
        if self.gpu == 0:
            self.writer.close()

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def adjust_learning_rate(self, epoch):
        """Decays the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            self.scheduler.step()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

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
        self.initial_m = args.m
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.gpu = args.gpu
        self.warmup_epochs = args.warmup_epochs
        self.scheduler = scheduler
        self.lr = args.lr
        self.accumulation_steps = args.accum
        self.print_freq = args.print_freq
        self.single = args.single
        if self.writer:
            _create_model_training_folder(self.writer, files_to_same=["main.py", 'trainer.py'])

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

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
        
        if not self.single:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        self.batch_size = int(self.batch_size / self.accumulation_steps)
        print(f"Adjusted batch size : {self.batch_size}")
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=(train_sampler is None),
                                  sampler = train_sampler)

        niter = 0
        if self.gpu == 0:
            model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):
            
            if not self.single:
                train_sampler.set_epoch(epoch_counter)

            lr = self.adjust_learning_rate(epoch_counter + 1)
            m = self.adjust_momentum(epoch_counter)
            if self.gpu == 0:
                self.writer.add_scalar('learning_rate', lr, global_step=epoch_counter)
                self.writer.add_scalar('momentum', m, global_step=epoch_counter)
            for i, ((batch_view_1, batch_view_2), _) in enumerate(train_loader):
                
                # print(f"input : {batch_view_1.norm()}")
                # print(f"input : {batch_view_2.norm()}")

                batch_view_1 = batch_view_1.to(self.gpu)
                batch_view_2 = batch_view_2.to(self.gpu)

                # if self.gpu == 0 and niter == 8:
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
                        # for params in list(self.online_network.parameters())[:3]:
                        #     print(f"i : {i}, param {params.sum()}")
                        #print(f"loss : {loss.item()}")
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_target_network_parameters()  # update the key encoder
                    if self.gpu == 0 and niter % self.print_freq == 0:
                        print("Loss on {}: {}".format(niter, (loss.item() * self.accumulation_steps)))
                    niter += 1
            print("End of epoch {}, loss : {}".format(epoch_counter, (loss.item() * self.accumulation_steps)))
            # save checkpoints
            if self.gpu == 0:
                self.save_model(os.path.join(model_checkpoints_folder, f'model{epoch_counter}.pth'))
        
        if self.gpu == 0:
            self.writer.close()

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        #print(list(self.online_network.children())[6:])

        # temp = batch_view_1.requires_grad_()
        # temp.register_hook(lambda grad : print(f"grad : {grad.sum().item()}"))
        # temp = batch_view_2.requires_grad_()
        # temp.register_hook(lambda grad : print(f"grad : {grad.sum().item()}"))
        
        # partition_number = 6

        # part1 = torch.nn.Sequential(*list(self.online_network.children())[:partition_number])
        # mid = part1(batch_view_1)
        # #temp = mid.requires_grad_()
        # #temp.register_hook(lambda grad : print(f"grad1 : {grad.norm().item()}"))
       
        # #print(part1)
        # #print(f"activation : {mid.norm()}")
        # part2 = torch.nn.Sequential(*list(self.online_network.children())[partition_number:])
        # mid = part2(mid)

        # prediction_partition_number = 1
        # part3 = torch.nn.Sequential(*list(self.predictor.children())[:prediction_partition_number])
        # mid = part3(mid)
        # #print(f"activation : {mid.norm()}")
        # part4 = torch.nn.Sequential(*list(self.predictor.children())[prediction_partition_number:])
        # predictions_from_view_1 = part4(mid)

        # mid = part1(batch_view_2)
        # #temp = mid.requires_grad_()
        # #temp.register_hook(lambda grad : print(f"grad2 : {grad.norm().item()}"))
        # #print(f"activation : {mid.norm()}")
        # mid = part2(mid)
        # #predictions_from_view_2 = self.predictor(mid)
        # mid = part3(mid)
        # #print(f"activation : {mid.norm()}")
        # predictions_from_view_2 = part4(mid)



        # partition_target_number = -4

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)
            # part1 = torch.nn.Sequential(*list(self.target_network.children())[:partition_target_number])
            # mid = part1(batch_view_1)
            # #print(f"activation : {mid.norm()}")
            # part2 = torch.nn.Sequential(*list(self.target_network.children())[partition_target_number:])
            # #print(part2)
            # targets_to_view_2 = part2(mid)

            # mid = part1(batch_view_2)
            # #print(f"activation : {mid.norm()}")
            # targets_to_view_1 = part2(mid)
        
        # detached_view_1 = predictions_from_view_1.requires_grad_()
        # detached_view_1.register_hook(lambda grad : print(f"grad1 : {grad.norm().item()}"))

        # detached_view_2 = predictions_from_view_2.requires_grad_()
        # detached_view_2.register_hook(lambda grad : print(f"grad2 : {grad.norm().item()}"))

        # print(f"predictions_from_view_1 : {predictions_from_view_1.norm()}")
        # print(f"predictions_from_view_2 : {predictions_from_view_2.norm()}")
        # print(f"targets_to_view_1 : {targets_to_view_1.norm()}")
        # print(f"targets_to_view_2 : {targets_to_view_2.norm()}")
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def adjust_learning_rate(self, epoch):
        """Decays the learning rate with half-cycle cosine after warmup"""
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.lr * 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.max_epochs + 1 - self.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def adjust_momentum(self, epoch):
        self.m = 1. - (1. - self.initial_m) * (1. + math.cos(math.pi * epoch / self.max_epochs)) * 0.5
        return self.m
    
    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

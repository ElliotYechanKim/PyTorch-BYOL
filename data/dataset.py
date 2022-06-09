import torch
from torchvision import transforms
from data.loader import TwoCropsTransform, GaussianBlur, Solarize
from torch.utils.data import Dataset
import numpy as np

class ProgressiveDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def increase_stage(self, epoch, writer=None, wandb=None):

        if self.args.interpolate == 'linear':
            s = self.args.init_prob + (self.args.max_prob - self.args.init_prob) / self.args.max_epochs * epoch
            
        elif self.args.interpolate == 'log':
            # s = np.exp(np.log(self.args.init_prob) + (np.log(self.args.max_prob - self.args.init_prob) - np.log(self.args.init_prob)) / \
            #                                                         np.log(self.args.num_stages) * np.log(epoch)) + self.args.init_prob
            s = np.exp(np.log(self.args.init_prob) + (np.log(self.args.max_prob) - np.log(self.args.init_prob)) / \
                                                                    np.log(self.args.max_epochs) * np.log(epoch + 1))

        #scale_lower = max((1 - s) + (0.08 - (1 - s)) / self.args.max_epochs * epoch, 0) #scale_lower = max(1 - s, 0.08)
        img_size = int(self.args.orig_img_size * s) #img_size = min(self.args.orig_img_size * s, self.args.orig_img_size)
        
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        normalize = transforms.Normalize(mean=mean, std=std)

        augmentation1 = [
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.)),
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
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.)),
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
        
        if writer:
            writer.add_scalar('s', s, global_step=epoch)
            writer.add_scalar('image size', img_size, global_step=epoch)
            #writer.add_scalar('scale_lower', scale_lower, global_step=epoch)
        
        if wandb:
            wandb.log({'s' : s})
            wandb.log({'image size' : img_size})
            #wandb.log({'scale_lower' : scale_lower})
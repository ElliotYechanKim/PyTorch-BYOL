import torch
import torchvision
from torchvision import datasets
from argparse import ArgumentParser
import torchvision.transforms as transforms
from data.loader import TwoCropsTransform
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

random_seed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(random_seed)
torch.manual_seed(random_seed)
import random
random.seed(random_seed)

parser = ArgumentParser()
parser.add_argument('--progressive', action='store_true')
args = parser.parse_args()

def main():

    writer = SummaryWriter()

    if not args.progressive:
        augmentation1 = [
            transforms.RandomResizedCrop(64, scale=(0.7, 0.7), ratio = (1, 1)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
        ]

        augmentation2 = [
            transforms.RandomResizedCrop(64, scale=(0.7, 0.7), ratio = (1, 1)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            # transforms.RandomApply([Solarize()], p=0.2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
        ]
        transforms_func = TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2))
    else:
        augmentation1 = [
            transforms.PILToTensor(),
        ]
        augmentation2 = [
            transforms.PILToTensor(),
        ]
        transforms_func = TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2))

    train_dataset =  torchvision.datasets.STL10('/home/ykim/data/stl10', split='train+unlabeled', transform=transforms_func)

    train_loader = DataLoader(train_dataset, batch_size = 4, num_workers = 8, 
                        drop_last = False, shuffle = True)

    for i, ((batch_view_1, batch_view_2), _) in enumerate(train_loader):
        if args.progressive:
            batch_view_1, batch_view_2 = adjust_augment_ratio(batch_view_1, batch_view_2)
        grid = torchvision.utils.make_grid(batch_view_1[:4])
        writer.add_image('views_1', grid, global_step=i)
        grid = torchvision.utils.make_grid(batch_view_2[:4])
        writer.add_image('views_2', grid, global_step=i)
        break

def adjust_augment_ratio(batch_view_1, batch_view_2):
        
        image_size = 96
        
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        normalize = transforms.Normalize(mean=mean, std=std)

        augmentation1 = transforms.Compose([
            np.ndarray,
            transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1 * s) 
            #], p=0.8),
            #transforms.RandomGrayscale(p=0.2 * s),
            #transforms.RandomApply([transforms.GaussianBlur(kernel_size = 9, sigma = [.1, 2.])], p=1.0),
            #transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            #transforms.RandomHorizontalFlip(p=0.5 * s),
            
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1 * s)
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2 * s),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0 * s),
            # transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float),
            #normalize
        ])

        augmentation2 = transforms.Compose([
            np.ndarray,
            transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
            #transforms.RandomApply([
            #    transforms.ColorJitter(0.4 * s, 0.4 * s, 0.2 * s, 0.1 * s)
            #], p=0.8),
            #transforms.RandomGrayscale(p=0.2 * s),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size = 9, sigma = [.1, 2.])], p=0.1),
            # transforms.RandomApply([transforms.RandomSolarize(threshold=128.0)], p=0.2 * s),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            # transforms.RandomApply([Solarize()], p=0.2),
            #transforms.RandomHorizontalFlip(p=0.5*s),
            transforms.ConvertImageDtype(torch.float),
            #normalize
        ])
        return augmentation1(batch_view_1), augmentation2(batch_view_2)

if __name__ == '__main__':
    main()

import torch
import torchvision
from argparse import ArgumentParser
import torchvision.transforms as transforms
from data.loader import TwoCropsTransform
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from similarity import SimFilter
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
import math

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
parser.add_argument('-b', '--batch-size', type=int, default=128)
parser.add_argument('--max-epochs', type=int, default=10)
parser.add_argument('--init-prob', type=float, default=0.5)
parser.add_argument('--max-prob', type=float, default=1.5)
parser.add_argument('--name', type=str, default='resnet18')
parser.add_argument('--filter-ratio', type=float, default=0.1)
parser.add_argument('--sim-pretrained', action='store_true', help = 'Using pre-trained model to masuer the similarity')
args = parser.parse_args()

class ProgressiveDataset(Dataset):
    def __init__(self, dataset):
        super(ProgressiveDataset, self).__init__()
        self.dataset = dataset
        self.increase_ratio(0)
        print('progressivedataset initialized')

    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)


    def increase_ratio(self, epoch):
        # if writer:
        #     writer.add_scalar('ratio', s, global_step=epoch)

        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        normalize = transforms.Normalize(mean=mean, std=std)

        augmentation1 = [
            transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize
        ]

        augmentation2 = [
            transforms.RandomResizedCrop(96, scale=(0.08, 1.)),
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            # transforms.RandomApply([Solarize()], p=0.2),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize
        ]
        transforms_func = TwoCropsTransform(transforms.Compose(augmentation1), transforms.Compose(augmentation2))
        self.dataset.transform = transforms_func

def similarity_test():

    writer = SummaryWriter()

    online_network = ResNet18(args.name)
    predictor = MLPHead(in_channels=online_network.projection.net[-1].out_features, name=args.name)
    train_dataset =  torchvision.datasets.STL10('/home/ykim/data/stl10', split='train+unlabeled')    
    print(len(train_dataset))
    if args.progressive:
        args.sigma3 = math.ceil(args.batch_size * 0.03)
        args.orig_batch_size = args.batch_size
        args.batch_size = int(args.batch_size / (1 - args.filter_ratio)) + 2 * args.sigma3

        orig_updates = len(train_dataset) / args.orig_batch_size
        print(orig_updates)
        updates = len(train_dataset) / args.batch_size
        print(updates)
        added_epochs = (orig_updates - updates) * args.max_epochs / updates
        print(added_epochs)
        simfilter = SimFilter(args, online_network)

    prog_train_dataset = ProgressiveDataset(train_dataset)
    train_loader = DataLoader(prog_train_dataset, batch_size = args.batch_size, num_workers = 8, 
                                            drop_last = False, shuffle = True)
        
    for epoch in range(args.max_epochs):
        for i, ((batch_view_1, batch_view_2), _) in enumerate(train_loader):
            print(batch_view_1.shape)
            if args.sim_pretrained:
                batch_view_1, batch_view_2 = simfilter.filter_by_similarity_ratio(batch_view_1, batch_view_2, epoch)
            else:
                batch_view_1, batch_view_2 = simfilter.filter_by_similarity_ratio(batch_view_1, batch_view_2, epoch)
            print(batch_view_1.shape)
        
        if epoch != args.max_epochs - 1:
            prog_train_dataset.increase_ratio(epoch + 1)
        break

def adjust_augment_ratio(batch_view_1, batch_view_2):
        
        image_size = 96
        
        mean = torch.tensor([0.43, 0.42, 0.39])
        std  = torch.tensor([0.27, 0.26, 0.27])
        normalize = transforms.Normalize(mean=mean, std=std)

        augmentation1 = transforms.Compose([
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
            transforms.ToTensor(),
            #normalize
        ])

        augmentation2 = transforms.Compose([
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
            transforms.ToTensor(),
            #normalize
        ])
        return augmentation1(batch_view_1), augmentation2(batch_view_2)

def interpolation_test():
    linear = []
    print('Linear interpolation TEST from 0 to 10')
    for i in range(args.max_epochs):
        s = args.init_prob + (args.max_prob - args.init_prob) / (args.max_epochs - 1) * i
        print(s)
    
    log_linear = []
    print('Log Linear interpolation TEST from 0 to 10')
    for i in range(args.max_epochs):
        s = np.exp(np.log(args.init_prob) + (np.log(args.max_prob) - np.log(args.init_prob)) / \
                                                        np.log(args.max_epochs) * np.log(i + 1))
        print(s)

def filter_length_tests():

    filter_ratio = 0.1
    orig_batch_size = 512
    reduce = int(orig_batch_size * filter_ratio)
    increased_batch = orig_batch_size + reduce
    
    begins = []
    ends = []
    for i in range(10):
        begin = int(reduce * i / 10)
        end = reduce - begin
        begins.append(begin)
        ends.append(end)

    print(begins)
    print(ends)

def scale_test():
    linear = []
    scale = []
    for i in range(10):
        s = args.init_prob + (1 - args.init_prob) / args.max_epochs * i
        scale_lower = max(1 - s, 0.08)
        linear.append(s)
        scale.append(scale_lower)
    print(linear)
    print(scale)

def inverse_linear_scale_test():
    scale = []
    ss = []
    linear_scale = []
    for i in range(args.max_epochs):
        s = args.init_prob + (args.max_prob - args.init_prob) / args.max_epochs * i
        scale_lower = max((1 - s) + (0.08 - (1 - s)) / args.max_epochs * i, 0)
        sscale = max((1 - args.init_prob) + (0.08 - (1 - args.init_prob)) / args.max_epochs * i, 0)
        scale.append(scale_lower)
        ss.append(s)
        linear_scale.append(sscale)
    
    print(ss)
    print(scale)
    print(linear_scale)

if __name__ == '__main__':
    interpolation_test()
    #similarity_test()
    #scale_test()
    #filter_length_tests()
    #inverse_linear_scale_test()
from utils.base import Config
from utils.image_aug import RandAugmentMC
from utils.dataset import TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Dataset,Subset
    
from torch.utils.data import DataLoader,random_split
import torchvision as tv
from .base import BaseDataset, TransformSubset
import torchvision.transforms as transforms




class Dataset(BaseDataset):

    def prepare_transforms(self):
        channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                            std = [0.2023, 0.1994, 0.2010])
        
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        
        
        
        self.eval_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(**channel_stats)
                ])
        aug_type=self.aug_type
        
        if aug_type[0]=='e':
            self.sup_transform = self.eval_transform
        elif aug_type[0]=='w':
            self.sup_transform = weak
        
        if aug_type[1]=='ww':
            self.unsup_transform = wstwice(weak, weak)
        elif aug_type[1]=='ws':
            self.unsup_transform = wstwice(weak, strong)
        elif aug_type[1]=='ew':
            self.unsup_transform = wstwice(self.eval_transform, weak)
        
    def prepare_data(self) -> None:
        if self.conf.dataset.num_classes==10:
            tv.datasets.CIFAR10(root=self.conf.dataset_dir, train=True, download=True)
            tv.datasets.CIFAR10(root=self.conf.dataset_dir, train=False, download=True)
            self.trainset_all = tv.datasets.CIFAR10(root=self.conf.dataset_dir, train=True, download=True)
            self.testset = tv.datasets.CIFAR10(root=self.conf.dataset_dir, train=False, download=True,transform=self.eval_transform)
            
        if self.conf.dataset.num_classes==100:
            tv.datasets.CIFAR100(root=self.conf.dataset_dir, train=True, download=True)
            tv.datasets.CIFAR100(root=self.conf.dataset_dir, train=False, download=True)
            self.trainset_all = tv.datasets.CIFAR100(root=self.conf.dataset_dir, train=True, download=True)
            self.testset = tv.datasets.CIFAR100(root=self.conf.dataset_dir, train=False, download=True,transform=self.eval_transform)
        

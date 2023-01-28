from utils.base import Config
from utils.image_aug import RandAugmentMC
from utils.dataset import TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Dataset,Subset
    
from torch.utils.data import DataLoader,random_split
import torchvision as tv
from .base import BaseDataset, TransformSubset,channel_stats,eval_transform
import torchvision.transforms as transforms




class Dataset(BaseDataset):
    def prepare_data(self) -> None:
        tv.datasets.CIFAR10(root=self.conf.dataset_dir, train=True, download=True)
        tv.datasets.CIFAR10(root=self.conf.dataset_dir, train=False, download=True)
    
    def setup(self, stage):
        weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])
        if stage=='fit':
            trainset_all = tv.datasets.CIFAR10(root=self.conf.dataset_dir, train=True, download=True)
            trainset,valset=uniform_split(trainset_all, [0.9,0.1])
            # split sup/unsup from trainset
            label_ratio=self.conf.label_ratio
            _supset,_=uniform_split(trainset, [label_ratio,1-label_ratio])
            self.supset=TransformSubset(_supset,weak)
            self.valset=TransformSubset(valset,eval_transform)
        if stage=='test':
            self.testset = tv.datasets.CIFAR10(root=self.conf.dataset_dir, train=False, download=True,transform=eval_transform)
            
    
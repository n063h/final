from utils.base import Config
from utils.image_aug import RandAugmentMC
from utils.dataset import TransformSubset, TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Dataset,Subset
    
from torch.utils.data import DataLoader,random_split
import torchvision as tv
from .base import channel_stats,eval_transform
from .cifar100 import Dataset as Cifar100
import torchvision.transforms as transforms


class Dataset(Cifar100):
    def setup(self,stage):
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
        unsup_transform = wstwice(weak, strong)
        
        
        if stage=='fit':
            trainset_all = tv.datasets.CIFAR100(root=self.conf.dataset_dir, train=True, download=True)
            trainset,valset=uniform_split(trainset_all, [0.9,0.1])
            # split sup/unsup from trainset
            label_ratio=self.conf.label_ratio
            _supset,_unsupset=uniform_split(trainset, [label_ratio,1-label_ratio])
            self.supset=TransformSubset(_supset,weak)
            self.unsupset=TransformSubset(_unsupset,unsup_transform)
            self.valset=TransformSubset(valset,eval_transform)
        if stage=='test':
            self.testset = tv.datasets.CIFAR100(root=self.conf.dataset_dir, train=False, download=True,transform=eval_transform)
        
    
    def train_dataloader(self):
        sup_loader=DataLoader(self.supset, batch_size=self.conf.sup_size, num_workers=self.num_workers, shuffle=True)
        unsup_loader=DataLoader(self.unsupset, batch_size=self.conf.unsup_size, num_workers=self.num_workers, shuffle=True)
        return [sup_loader,unsup_loader]
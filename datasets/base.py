
from utils.base import Config

from utils.dataset import TransformSubset, TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split

from torch.utils.data import DataLoader,Dataset,Subset
    
from torch.utils.data import DataLoader,random_split
import torchvision as tv

import torchvision.transforms as transforms






class BaseDataset():
    def __init__(self,conf:Config) -> None:
        super().__init__()
        self.conf=conf
        self.num_workers=conf.num_workers
        self.num_classes=conf.dataset.num_classes
        self.trainset_all=[]
        self.testset_all = []
        self.sup_transform = None
        self.unsup_transform = None
        self.eval_transform = None
        self.aug_type=(self.conf.arch.aug_type or 'e_ww').split('_')
        assert len(self.aug_type)==2
        
    def prepare_data(self) -> None:
        pass
    
    def prepare_transforms(self):
        pass
    
    
    def setup(self, stage):
        if stage=='fit':
            trainset,valset=uniform_split(self.trainset_all, [0.8,0.2])
            # split sup/unsup from trainset
            label_ratio=self.conf.label_ratio
            _supset,_unsupset=uniform_split(trainset, [label_ratio,1-label_ratio])
            self.supset=TransformSubset(_supset,self.sup_transform)
            self.unsupset=TransformSubset(trainset,self.unsup_transform)
            self.valset=TransformSubset(valset,self.eval_transform)
        if stage=='test':
            self.testset = TransformSubset(self.testset_all,self.eval_transform)
        
    
    def train_dataloader(self):
        sup_loader=DataLoader(self.supset, batch_size=self.conf.dataset.sup_size, num_workers=self.num_workers, shuffle=True)
        unsup_loader=DataLoader(self.unsupset, batch_size=self.conf.dataset.unsup_size, num_workers=self.num_workers, shuffle=True)
        return [sup_loader,unsup_loader]

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.conf.dataset.eval_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.conf.dataset.eval_size, num_workers=self.num_workers, shuffle=False)

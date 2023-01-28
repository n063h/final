
from utils.base import Config

from utils.dataset import TransformSubset, TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split

from torch.utils.data import DataLoader,Dataset,Subset
    
from torch.utils.data import DataLoader,random_split
import torchvision as tv

import torchvision.transforms as transforms




channel_stats = dict(mean = [0.4914, 0.4822, 0.4465],
                            std = [0.2023, 0.1994, 0.2010])
eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])

class BaseDataset():
    def __init__(self,conf:Config) -> None:
        super().__init__()
        self.conf=conf
        self.num_workers=conf.num_workers
        
    def prepare_data(self) -> None:
        pass
    
    def setup(self, stage):
        if stage=='fit':
            trainset_all = []
            trainset,valset=uniform_split(trainset_all, [0.9,0.1])
            # split sup/unsup from trainset
            label_ratio=self.conf.label_ratio
            _supset,_=uniform_split(trainset, [label_ratio,1-label_ratio])
            self.supset=TransformSubset(_supset,None)
            self.valset=TransformSubset(valset,None)
        if stage=='test':
            self.testset = []
        
    
    def train_dataloader(self):
        return DataLoader(self.supset, batch_size=self.conf.dataset.sup_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.conf.dataset.eval_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.conf.dataset.eval_size, num_workers=self.num_workers, shuffle=False)

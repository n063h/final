from utils.base import Config
from utils.image_aug import RandAugmentMC
from utils.dataset import MyDataset, TransformSubset, TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Dataset,Subset,BatchSampler,RandomSampler
    
from torch.utils.data import DataLoader

from utils.nir_dataset import RandAugmentNir, ToTensor, read_npy
from .cls50 import Dataset as Cls50
import torchvision.transforms as transforms


class Dataset(Cls50):
    def setup(self,stage):
        weak = transforms.Compose([
            ToTensor(),
            RandAugmentNir(choice_num=1, max_value_ratio=0.2)
        ])
        strong = transforms.Compose([
            ToTensor(),
            RandAugmentNir(choice_num=2, max_value_ratio=1)
        ])
        unsup_transform = wstwice(weak, strong)
        eval_transform = transforms.Compose([
            ToTensor(),
        ])
        
        if stage=='fit':
            data,targets=read_npy(self.conf.dataset_dir+'/train')
            if (axis:=self.conf.axis) is not None:
                data=data[:,axis,:]
            trainset_all = MyDataset(data,targets)
            trainset,valset=uniform_split(trainset_all, [0.9,0.1])
            # split sup/unsup from trainset
            label_ratio=self.conf.label_ratio
            _supset,_unsupset=uniform_split(trainset, [label_ratio,1-label_ratio])
            self.supset=TransformSubset(_supset,weak)
            self.unsupset=TransformSubset(_unsupset,unsup_transform)
            self.valset=TransformSubset(valset,eval_transform)
        if stage=='test':
            data,targets=read_npy(self.conf.dataset_dir+'/test')
            self.testset = MyDataset(data,targets,eval_transform)
        
    
    def train_dataloader(self):
        sup_sampler=BatchSampler(RandomSampler(self.supset),batch_size=self.conf.sup_size)
        sup_loader=DataLoader(self.supset, batch_size=self.conf.sup_size, num_workers=self.num_workers, batch_sampler=sup_sampler)
        unsup_sampler=BatchSampler(RandomSampler(self.unsupset),batch_size=self.conf.unsup_size)
        unsup_loader=DataLoader(self.unsupset, batch_size=self.conf.unsup_size, num_workers=self.num_workers,batch_sampler=unsup_sampler)
        return [sup_loader,unsup_loader]
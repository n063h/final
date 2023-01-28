from utils.base import Config
from utils.image_aug import RandAugmentMC
from utils.dataset import MyDataset, TransformSubset, TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Dataset,Subset,BatchSampler,RandomSampler
    
from torch.utils.data import DataLoader

from utils.nir_dataset import RandAugmentNir, ToTensor, read_npy
from .cls10 import Dataset as Cls10
import torchvision.transforms as transforms

# for fixmatch
class Dataset(Cls10):
    def setup(self,stage):
        weak = transforms.Compose([
            ToTensor(),
            *self.conf.dataset.w_augs
        ])
        strong = transforms.Compose([
            ToTensor(),
            *self.conf.dataset.s_augs
        ])
        unsup_transform = wstwice(weak, strong)
        eval_transform = transforms.Compose([
            ToTensor(),
        ])
        
        if stage=='fit':
            data,targets=read_npy(self.conf.dataset_dir+'/cls10_train')
            if (axis:=self.conf.dataset.axis) is not None:
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
            data,targets=read_npy(self.conf.dataset_dir+'/cls10_test')
            self.testset = MyDataset(data,targets,eval_transform)
        
    
    def train_dataloader(self):
        sup_sampler=RandomSampler(self.supset)
        sup_loader=DataLoader(self.supset, batch_size=self.conf.dataset.sup_size, num_workers=self.num_workers, sampler=sup_sampler)
        unsup_sampler=RandomSampler(self.unsupset)
        unsup_loader=DataLoader(self.unsupset, batch_size=self.conf.dataset.unsup_size, num_workers=self.num_workers,sampler=unsup_sampler)
        return [sup_loader,unsup_loader]
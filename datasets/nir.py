from utils.base import Config
from utils.image_aug import RandAugmentMC
from utils.dataset import MyDataset, TransformEachDim, TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Dataset,Subset
    
from torch.utils.data import DataLoader,random_split
import torchvision as tv

from utils.nir_dataset import ToTensor, read_npy
from .base import BaseDataset, TransformSubset
import torchvision.transforms as transforms




class Dataset(BaseDataset):

    def prepare_transforms(self):
        
        weak=TransformEachDim(ToTensor(),self.conf.dataset.w_augs,self.conf.dataset.dim_aug)
        strong=TransformEachDim(ToTensor(),self.conf.dataset.w_augs,self.conf.dataset.dim_aug)
            
        self.eval_transform = transforms.Compose([
            ToTensor(),
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
        get_file=lambda t:f'{self.conf.dataset_dir}/cls{self.conf.dataset.num_classes}_{t}'
    
        data,targets=read_npy(get_file('train'))
        if (axis:=self.conf.dataset.axis) is not None:
            data=data[:,axis,:].reshape(data.shape[0],-1,data.shape[-1])
        self.trainset_all = MyDataset(data,targets)
        
        data,targets=read_npy(get_file('test'))
        if (axis:=self.conf.dataset.axis) is not None:
            data=data[:,axis,:].reshape(data.shape[0],-1,data.shape[-1])
        self.testset = MyDataset(data,targets,self.eval_transform)
        

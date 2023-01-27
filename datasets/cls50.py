from utils.base import Config
from utils.image_aug import RandAugmentMC
from utils.dataset import MyDataset, TransformSubset, TransformWeakStrong as wstwice,TransformBaseWeakStrong as bwstwice, uniform_split
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader,Dataset,Subset
    
from torch.utils.data import DataLoader,random_split

from utils.nir_dataset import RandAugmentNir, ToTensor, read_npy
from .base import BaseDataset
import torchvision.transforms as transforms


class Dataset(BaseDataset):
    def setup(self,stage):
        eval_transform = transforms.Compose([
            ToTensor(),
        ])
        if stage=='fit':
            data,targets=read_npy(self.conf.dataset_dir+'/cls50_train')
            if (axis:=self.conf.axis) is not None:
                data=data[:,axis,:]
            trainset_all = MyDataset(data,targets)
            trainset,valset=uniform_split(trainset_all, [0.9,0.1])
            self.trainset=TransformSubset(trainset,eval_transform)
            self.valset=TransformSubset(valset,eval_transform)
        if stage=='test':
            data,targets=read_npy(self.conf.dataset_dir+'/cls50_test')
            self.testset = MyDataset(data,targets,eval_transform)
        
    
    def train_dataloader(self):
        train_loader=DataLoader(self.trainset, batch_size=self.conf.sup_size, num_workers=self.num_workers, shuffle=True)
        return train_loader
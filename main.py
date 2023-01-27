from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback
import os,shutil
from torch import nn
from importlib import import_module

from dataclasses import dataclass
import warnings
from utils.base import Config
from utils.nir_aug import *
from utils.device import get_device
seed_everything(42)
warnings.filterwarnings("ignore")

def main(conf:Config):
    dash=lambda :print("-------------------------------------------------------------------------")
    dash()
    print(vars(conf))
    dash()
    
    

    device=get_device()
    dataset:LightningDataModule=import_module('datasets.'+conf.dataset).Dataset(conf)
    model:nn.Module=import_module('models.'+conf.model).Net(conf)
    arch=import_module('arch.'+conf.arch).Arch(model=model,conf=conf)
    trainer = Trainer(logger=tb_logger,callbacks=callbacks,default_root_dir=conf.default_root_dir,
                      min_epochs=conf.min_epochs,max_epochs=conf.max_epochs,**device)
    trainer.fit(arch, datamodule=dataset)
    
if __name__ == '__main__':
    confs=[
        # Config(name='fixmatch_cls10ew',semi=True,dataset='cls10ew',model='resnet1d',arch='fixmatch',label_ratio=0.01,sup_size=1024,unsup_size=1024),
        # Config(name='fixmatch_cls10ws',semi=True,dataset='cls10ws',model='resnet1d',arch='fixmatch',label_ratio=0.01,sup_size=1024,unsup_size=1024),
        # Config(name='pi_cls10ww',semi=True,dataset='cls10ww',model='resnet1d',arch='pi',label_ratio=0.01,sup_size=1024,unsup_size=1024),
        # Config(name='nosemi_cifar10',semi=False,dataset='cls10',model='resnet1d',arch='nosemi',sup_size=1024,unsup_size=1024),
        Config(name='tri_cls10',semi=False,dataset='cls10',model='3resnet1d',arch='tri',sup_size=1024,unsup_size=1024),
    ]
    augs={
        # 'jitter':[(DA_Jitter,0.05,0),(DA_Jitter,0.02,0),(DA_Jitter,0.10,0),(DA_Jitter,0.05,0.05)],
        # 'scaling':[(DA_Scaling,0.1,1),(DA_Scaling,0.05,1),(DA_Scaling,0.2,1),(DA_Scaling,0.1,0.5)],
        'scaling':[(DA_Scaling,0.1,1),(DA_Scaling,0.2,1)],
        # 'magwarp':[(DA_MagWarp,0.05,0),(DA_MagWarp,0.02,0),(DA_MagWarp,0.10,0),(DA_MagWarp,0.05,0.05)],
        'magwarp':[(DA_MagWarp,0.02,0),(DA_MagWarp,0.10,0)],
    }
    # shutil.rmtree(confs[0].log_dir)
    for conf in confs:
        for augname,group_aug in augs.items(): # every kind of aug
            for (augfunc,alpha,bate) in group_aug: # every aug testcase
                name=conf.name
                conf.name=conf.name+f"_{augname}_{alpha}_{bate}"
                conf.w_augs=[BaseTransform(augfunc,alpha,bate)]
                conf.s_augs=[BaseTransform(augfunc,alpha,bate)]
                main(conf)
                try:
                    main(conf)
                except Exception as e:
                    conf.name=name
    os.system("shutdown")
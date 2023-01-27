from dataclasses import dataclass
from datetime import datetime

@dataclass
class Config:
    # common config
    num_classes:int=10
    num_workers:int=1
    axis:int=None
    w_augs=[]
    s_augs=[]
    
    # custom config
    name:str='test'
    sup_size:int=1024
    unsup_size:int=1024
    eval_size:int=1000
    dataset:str='cls10'
    model:str='resnet1d'
    arch:str='fixmatch'
    usp_weight:float=1
    label_ratio:float=0.1
    min_epochs:int=10
    max_epochs:int=20
    lr:float=0.1
    threshold:float=0.95
    semi:bool=True
    
    # lightning config
    default_root_dir:str='/home/lvhang/autodl-tmp/checkpoints'
    log_dir:str='/home/lvhang/tf-logs'
    dataset_dir:str='/home/lvhang/autodl-tmp/datasets'
    
class Timer:
    def __init__(self) -> None:
        self.timer=datetime.now()
    
    def update(self):
        cur=datetime.now()
        diff=cur-self.timer
        self.timer=cur
        return diff
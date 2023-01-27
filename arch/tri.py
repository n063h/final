from datetime import datetime
import torch
from  torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from utils.base import Config, Timer
from utils.ema import EMA
from utils.metrics import get_multiclass_acc_metrics


class Arch(LightningModule):
    def __init__(self,model,conf:Config,*args,**kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.models=model
        self.conf=conf
        
        metrics=get_multiclass_acc_metrics(conf.num_classes)
        self.train_metrics=[metrics.clone(prefix=f'train{i}_') for i in range(3)]
        self.val_metrics=[metrics.clone(prefix=f'val{i}_') for i in range(3)]
        self.timer=Timer()
        # self.automatic_optimization = False
        
    def forward(self,x):
        m1,m2,m3=self.models
        x1,x2,x3=x[:,0,:],x[:,1,:],x[:,2,:]
        return m1(x1),m2(x2),m3(x3)
    
    def on_train_epoch_start(self) -> None:
        self.timer.update()
    
    def training_step(self, batch, batch_idx,optimizer_idx):
        x, y = batch
        p = self.forward(x)
        loss=[]
        opts=self.optimizers()
        for i in range(3):
            li = F.cross_entropy(p[i], y)
            self.log_dict({f'train{i}_loss':li},prog_bar=True,on_step=True)
            loss.append(li)
            # opts[i].zero_grad()
            # self.manual_backward(li)
            # opts[i].step()
        
        return {'loss':loss,'pred':p,'y':y}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pred,y=outputs['pred'],outputs['y']
        for i in range(3):
            pi,mi=pred[i],self.train_metrics[i]
            metrics=mi(pi,y)
            self.log_dict(metrics,prog_bar=True,on_step=True)
        
    def on_train_epoch_end(self) -> None:
        diff=self.timer.update() 
        print("train_epoch_time",diff)
    
    def configure_optimizers(self):
        lr=0.01 if 'cls' in self.conf.dataset else 0.1
        m1,m2,m3=self.models
        o1 = torch.optim.SGD(m1.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
        o2 = torch.optim.SGD(m2.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
        o3 = torch.optim.SGD(m3.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
        l1 = torch.optim.lr_scheduler.CosineAnnealingLR(o1, T_max=self.conf.max_epochs,eta_min=1e-4)
        l2 = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=self.conf.max_epochs,eta_min=1e-4)
        l3 = torch.optim.lr_scheduler.CosineAnnealingLR(o3, T_max=self.conf.max_epochs,eta_min=1e-4)
        return [o1,o2,o3], [l1,l2,l3]

    def on_validation_epoch_start(self) -> None:
        self.timer.update()
        for i in range(3):
            self.val_metrics[i].reset()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        for i in range(3):
            li=F.cross_entropy(pred[i], y)
            self.val_metrics[i].update(pred[i], y)
            self.log(f'val{i}_loss',li,prog_bar=True,on_step=True)
        
    def on_validation_epoch_end(self) -> None:
        diff=self.timer.update()
        print("val_epoch_time",diff)
        for i in range(3):
            metrics=self.val_metrics[i].compute()
            self.log_dict(metrics,prog_bar=True,on_epoch=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        p = self.forward(x)
        for i in range(3):
            li = F.cross_entropy(p[i],y)
            self.log_dict({f'test{i}_loss':li},prog_bar=True,on_step=True)
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        return pred
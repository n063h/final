from datetime import datetime
import torch
from  torch import nn
import torch.nn.functional as F
import wandb
from arch.base import BaseModel
from utils.base import Config, Timer
from utils.ema import EMA
from utils.metrics import get_multiclass_acc_metrics


class Arch(BaseModel):
    def __init__(self,model:nn.Module,conf:Config,device,*args,**kwargs) -> None:
        self.models=model
        self.conf=conf
        self.device=device
        
        metrics=get_multiclass_acc_metrics(conf.dataset.num_classes,device)
        self.train_metrics=[metrics.clone(prefix=f'train{i}_') for i in range(3)]
        self.val_metrics=[metrics.clone(prefix=f'val{i}_') for i in range(3)]
        self.test_metrics=[metrics.clone(prefix=f'test{i}_') for i in range(3)]
        self.timer=Timer()
        self.log=print if conf.name=='test_train' else wandb.log
        self.h=torch.zeros(3,conf.dataset.num_classes).to(device)
        self.hc=torch.zeros(3,conf.dataset.num_classes).to(device)
        self.h_last=torch.zeros(3,conf.dataset.num_classes).to(device)
        
    def forward(self,x):
        m1,m2,m3=self.models
        x1,x2,x3=x[:,0,:],x[:,1,:],x[:,2,:]
        # x1,x2,x3=x[:,0,:],x[:,0,:],x[:,0,:]
        return m1(x1),m2(x2),m3(x3)
    
    def on_train_epoch_start(self,epoch) -> None:
        print("——————第 {} 轮训练开始——————".format(epoch + 1))
        for m in self.models:
            m.train()
        self.timer.update()
    
    def training_step(self, batch, batch_idx,optimizers):
        (x, y),_ = batch
        device=self.device
        x,y=x.to(device),y.to(device)
        p = self.forward(x)
        for i in range(3):
            li = F.cross_entropy(p[i], y)
            self.log({f'train{i}_loss':li.item()})
            optimizers[i].zero_grad()
            li.backward()
            optimizers[i].step()
        
        return {'loss':[],'pred':p,'y':y}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pred,y=outputs['pred'],outputs['y']
        for i in range(3):
            pi,mi=pred[i],self.train_metrics[i]
            metrics=mi(pi,y)
            self.log(metrics)
        self.h_last=self.h/self.hc
        self.h_last[self.h_last.isnan()]=0
        self.h=torch.zeros_like(self.h).to(self.device)
        self.hc=torch.zeros_like(self.hc).to(self.device)
        
    
    def configure_optimizers(self):
        lr=self.conf.dataset.lr
        m1,m2,m3=self.models
        o1 = torch.optim.SGD(m1.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
        o2 = torch.optim.SGD(m2.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
        o3 = torch.optim.SGD(m3.parameters(), lr=lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
        l1 = torch.optim.lr_scheduler.CosineAnnealingLR(o1, T_max=self.conf.max_epochs,eta_min=1e-4)
        l2 = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=self.conf.max_epochs,eta_min=1e-4)
        l3 = torch.optim.lr_scheduler.CosineAnnealingLR(o3, T_max=self.conf.max_epochs,eta_min=1e-4)
        self.optimizers=[o1,o2,o3]
        self.lr_schedulers=[l1,l2,l3]

    def on_validation_epoch_start(self,epoch) -> None:
        self.timer.update()
        print("——————第 {} 轮验证开始——————".format(epoch + 1))
        for i in range(3):
            self.val_metrics[i].reset()
            self.models[i].eval()
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device=self.device
        x,y=x.to(device),y.to(device)
        pred = self.forward(x)
        for i in range(3):
            li=F.cross_entropy(pred[i], y)
            self.val_metrics[i].update(pred[i], y)
            self.log({f'val{i}_loss':li})
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        diff=self.timer.update()
        print("val_epoch_time",diff)
        m=[]
        for i in range(3):
            metrics=self.val_metrics[i].compute()
            self.log(metrics)
            m.append(metrics[f"val{i}_acc"].item())
        print(m)
        return max(m)
        
        
    @torch.no_grad()
    def on_test_start(self):
        print("——————测试开始——————")
        self.timer.update()
        for metrics,model in zip(self.test_metrics,self.models):
            metrics.reset()
            model.eval()
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        device=self.device
        x,y=x.to(device),y.to(device)
        pred = self.forward(x)
        for m,p in zip(self.test_metrics,pred):
            m.update(p,y)
        
    @torch.no_grad()
    def on_test_end(self):
        diff=self.timer.update()
        print("test_time",diff)
        for m in self.test_metrics:
            metrics=m.compute()
            self.log(metrics)
            print(metrics)
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        return pred
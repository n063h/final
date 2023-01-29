from datetime import datetime
import torch
from  torch import nn
import torch.nn.functional as F
import wandb
from arch.tri import Arch as _Arch
from utils.base import Config, Timer
from utils.ema import EMA
from utils.metrics import get_multiclass_acc_metrics
from utils.ramps import exp_rampup


rampup=exp_rampup(30)
# pi + mixed_prob psudo-label
class Arch(_Arch):
    def __init__(self,model:nn.Module,conf:Config,device,*args,**kwargs) -> None:
        self.models=model
        self.conf=conf
        self.device=device
        
        metrics=get_multiclass_acc_metrics(conf.dataset.num_classes,device)
        self.train_metrics=[metrics.clone(prefix=f'train{i}_') for i in range(3)]
        self.val_metrics=[metrics.clone(prefix=f'val{i}_') for i in range(3)]
        self.test_metrics=[metrics.clone(prefix=f'test{i}_') for i in range(3)]
        self.timer=Timer()
        
    def forward(self,x):
        m1,m2,m3=self.models
        x1,x2,x3=x[:,0,:],x[:,1,:],x[:,2,:]
        return m1(x1),m2(x2),m3(x3)
    
    def on_train_epoch_start(self,epoch) -> None:
        print("——————第 {} 轮训练开始——————".format(epoch + 1))
        for m in self.models:
            m.train()
        self.timer.update()
        
        if epoch>0:
            self.h_last=self.h/self.hc
        else:
            self.h_last=torch.zeros((3,self.conf.dataset.num_classes)).to(self.device)
        self.h=torch.zeros((3,self.conf.dataset.num_classes)).to(self.device)
        self.hc=torch.zeros((3,self.conf.dataset.num_classes)).to(self.device)
    
    
    
    def training_step(self, batch, batch_idx,optimizers):
        ((sup_x1,sup_x2),sup_y), ((unsup_x1,unsup_x2),unsup_y) = batch
        device=self.device
        sup_x1,sup_x2,sup_y=sup_x1.to(device),sup_x2.to(device),sup_y.to(device)
        
        p10,p11,p12 = self.forward(sup_x1)
        # l0,l1,l2 are the loss of ith model, currently CE
        l0,l1,l2 = F.cross_entropy(p10, sup_y),F.cross_entropy(p11, sup_y),F.cross_entropy(p12, sup_y)
        h0,h1,h2 = F.softmax(p10, dim=1),F.softmax(p11, dim=1),F.softmax(p12, dim=1)
        prob0,label0=torch.max(h0,dim=1)
        prob1,label1=torch.max(h1,dim=1)
        prob2,label2=torch.max(h2,dim=1)
        
        # update epoch supervised hx and cnt by class
        for idx,prob,label in zip([0,1,2],[prob0,prob1,prob2],[label0,label1,label2]):
            for i in range(self.conf.dataset.num_classes):
                mask=torch.where(label==i)[0]
                self.h[idx,i]+=prob[mask].sum()
                self.hc[idx,i]+=len(mask)
                
        if self.conf.semi:
            unsup_x1,unsup_x2,unsup_y=unsup_x1.to(device),unsup_x2.to(device),unsup_y.to(device)
            with torch.no_grad():
                p20,p21,p22 = self.forward(sup_x2)
                up10,up11,up12 = self.forward(unsup_x1)
                up20,up21,up22 = self.forward(unsup_x2)
                up10,up11,up12,up20,up21,up22=up10.detach(),up11.detach(),up12.detach(),up20.detach(),up21.detach(),up22.detach()
            
            # smse0,smse1,smse2 are mse loss of sup_x1,sup_x2, umse0,umse1,umse2 are mse loss of unsup_x1,unsup_x2
            r=rampup(self.current_epoch)
            smse0,smse1,smse2=F.mse_loss(p10,p20)*r,F.mse_loss(p11,p21)*r,F.mse_loss(p12,p22)*r
            umse0,umse1,umse2=F.mse_loss(up10,up20)*r,F.mse_loss(up11,up21)*r,F.mse_loss(up12,up22)*r
            mse0,mse1,mse2=smse0+umse0,smse1+umse1,smse2+umse2
            
            # current model dominated by other 2 models 4 mean hx 
            uh10,uh11,uh12,uh20,uh21,uh22 =F.softmax(up10,dim=1),F.softmax(up11,dim=1),F.softmax(up12,dim=1),F.softmax(up20,dim=1),F.softmax(up21,dim=1),F.softmax(up22,dim=1)
            uprob0,ulabel0=torch.max((uh11+uh21+uh12+uh22)/4,dim=1)
            uprob1,ulabel1=torch.max((uh10+uh20+uh12+uh22)/4,dim=1)
            uprob2,ulabel2=torch.max((uh10+uh20+uh11+uh21)/4,dim=1)
            
            mask0 = uprob0.ge(self.conf.arch.threshold).float()
            mask1 = uprob1.ge(self.conf.arch.threshold).float()
            mask2 = uprob2.ge(self.conf.arch.threshold).float()
            
            # 随便选ww第一个
            ce0=torch.mean(F.cross_entropy(up10,ulabel0,reduction='none')*mask0)
            ce1=torch.mean(F.cross_entropy(up11,ulabel1,reduction='none')*mask1)
            ce2=torch.mean(F.cross_entropy(up12,ulabel2,reduction='none')*mask2)
            
            l0+=mse0+ce0
            l1+=mse1+ce1
            l2+=mse2+ce2
        
        loss=[l0,l1,l2]
        for i,l in zip(range(3),loss):
            wandb.log({f'train{i}_loss':l})
            optimizers[i].zero_grad()
            l.backward()
            optimizers[i].step()
                
        
        return {'loss':loss,'pred':[p10,p11,p12],'y':sup_y}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pred,y=outputs['pred'],outputs['y']
        for i in range(3):
            pi,mi=pred[i],self.train_metrics[i]
            metrics=mi(pi,y)
            wandb.log(metrics)
        
    
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

    @torch.no_grad()
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
            wandb.log({f'val{i}_loss':li})
    
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        diff=self.timer.update()
        print("val_epoch_time",diff)
        m=0
        for i in range(3):
            metrics=self.val_metrics[i].compute()
            wandb.log(metrics)
            m=max(m,metrics[f"val{i}_acc"].item())
        return m
        
        
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
            wandb.log(metrics)
            print(m)
        
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        return pred
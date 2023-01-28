from datetime import datetime
from itertools import cycle
import torch,os
from  torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule
import wandb
from utils.base import Config, Timer
from utils.ema import EMA
from utils.metrics import get_multiclass_acc_metrics
from tqdm import tqdm
from tqdm.contrib import tzip,tenumerate

class BaseModel():
    def __init__(self,model:nn.Module,conf:Config,device,*args,**kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model=model
        self.ema=EMA(self.model,0.97,device)
        self.ema.register()
        self.conf=conf
        self.device=device
        
        metrics=get_multiclass_acc_metrics(conf.dataset.num_classes,device)
        self.train_metrics=metrics.clone(prefix='train_')
        self.val_metrics=metrics.clone(prefix='val_')
        self.ema_metrics=metrics.clone(prefix='ema_')
        
        self.timer=Timer()
    
    def fit(self,dataset:LightningDataModule):
        dataset.prepare_data()
        dataset.setup('fit')
        train_dataloader=dataset.train_dataloader()
        val_dataloader=dataset.val_dataloader()
        conf=self.conf
        self.configure_optimizers()
        optimizers,lr_schedulers=self.optimizers,self.lr_schedulers
        
        if conf.resume:
            checkpoint = torch.load(os.path.join(conf.default_root_dir,conf.name))
            m,o,l=checkpoint['models'],checkpoint['optimizers'],checkpoint['lr_schedulers']
            if hasattr(self,'model'):
                self.model.load_state_dict(m[0])
            else:
                for mo,state in zip(self.models,m):
                    mo.load_state_dict(state)
            for opt,state in zip(optimizers,o):
                opt.load_state_dict(state)
            for ls,state in zip(lr_schedulers,l):
                ls.load_state_dict(state)
                
        self.best={'val_acc':0,'epoch':0}
        for epoch in range(conf.max_epochs):
            self.current_epoch=epoch
            
            self.on_train_epoch_start(epoch)
            self.training_epoch(train_dataloader,optimizers,epoch) # forward, loss , optimizer
            self.on_train_epoch_end(lr_schedulers) # lr_scheduler
            
            
            
            with torch.no_grad():
                self.on_validation_epoch_start(epoch)
                self.validation_epoch(val_dataloader,epoch) # forward, loss , optimizer
                val_acc=self.on_validation_epoch_end() # lr_scheduler
                
                if val_acc>self.best['val_acc']:
                    self.best={
                        'val_acc':val_acc,
                        'epoch':self.current_epoch
                    }
                    if hasattr(self,'model'):
                        self.models=[self.model]
                    state={'models':[m.state_dict() for m in self.models],'optimizers':[o.state_dict() for o in self.optimizers],'lr_schedulers':[l.state_dict() for l in lr_schedulers]}
                    torch.save(state, os.path.join(conf.default_root_dir,conf.name))
                print("best",self.best)
            
    def training_epoch(self,train_dataloader,optimizers,epoch):
        if not isinstance(train_dataloader,list):
            for idx,batch in tenumerate(train_dataloader, total =len(train_dataloader)):
                output=self.training_step(batch,idx,optimizers)
                self.on_train_batch_end(output,batch,idx)
        else:
            sup_loader,unlab_loader=train_dataloader
            idx=0
            for batch in tzip(cycle(sup_loader), unlab_loader, total =len(unlab_loader)):
                output=self.training_step(batch,idx,optimizers)
                self.on_train_batch_end(output,batch,idx)
                idx+=1
        
        
    def validation_epoch(self,val_dataloader,epoch):
        for idx,batch in tenumerate(val_dataloader, total =len(val_dataloader)):
                self.validation_step(batch,idx)
        
    def on_train_epoch_start(self,epoch) -> None:
        print("——————第 {} 轮训练开始——————".format(epoch + 1))
        self.model.train()
        self.timer.update()
    
    def training_step(self, batch, batch_idx,optimizers):
        x, y = batch
        optimizer=optimizers[0]
        device=self.device
        x,y=x.to(device),y.to(device)
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({'train_loss':loss})
        return {'loss':loss,'pred':pred,'y':y}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pred,y=outputs['pred'],outputs['y']
        metrics=self.train_metrics(pred,y)
        wandb.log(metrics)
        self.ema.update()
        
    def on_train_epoch_end(self,lr_schedulers) -> None:
        diff=self.timer.update()
        for lr in lr_schedulers:
            lr.step()
        print("train_epoch_time",diff)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.conf.dataset.lr,momentum=0.9,weight_decay=5e-4,nesterov=True)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.conf.max_epochs,eta_min=1e-4)
        self.optimizers=[optimizer]
        self.lr_schedulers=[lr_scheduler]

    @torch.no_grad()
    def on_validation_epoch_start(self,epoch) -> None:
        self.timer.update()
        self.model.eval()
        self.val_metrics.reset()
        self.ema_metrics.reset()
        print("——————第 {} 轮测试开始——————".format(epoch + 1))
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        device=self.device
        x,y=x.to(device),y.to(device)
        pred = self.model(x)
        val_loss = F.cross_entropy(pred, y)
        self.val_metrics.update(pred,y)
        wandb.log({'val_loss':val_loss})
        
        self.ema.apply_shadow()
        ema_pred = self.model(x)
        ema_loss = F.cross_entropy(ema_pred, y)
        self.ema_metrics.update(ema_pred,y)
        self.ema.restore()
        wandb.log({'ema_loss':ema_loss})
        
    @torch.no_grad()
    def on_validation_epoch_end(self) -> None:
        diff=self.timer.update()
        print("val_epoch_time",diff)
        metrics=self.val_metrics.compute()
        ema_metrics=self.ema_metrics.compute()
        wandb.log(metrics)
        wandb.log(ema_metrics)
        print(metrics,ema_metrics,end='')
        
        return metrics['val_acc'].item()

        
        
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        device=self.device
        x,y=x.to(device),y.to(device)
        pred = self.model(x)
        test_loss = F.cross_entropy(pred, y)
        
        self.ema.apply_shadow()
        ema_pred = self.model(x)
        ema_loss = F.cross_entropy(ema_pred, y)
        self.ema.restore()
        wandb.log({"test_loss": test_loss, "ema_test_loss": ema_loss})
    
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        x, y = batch
        device=self.device
        x,y=x.to(device),y.to(device)
        pred = self.model(x)
        return pred
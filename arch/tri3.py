from datetime import datetime
import torch
from  torch import nn
import torch.nn.functional as F
import wandb
from arch.tri1 import Arch as _Arch
from utils.base import Config, Timer
from utils.ema import EMA
from utils.metrics import get_multiclass_acc_metrics
from utils.ramps import exp_rampup


rampup=exp_rampup(40)
# pi + mixed_prob psudo-label + epoch h confidence by class
class Arch(_Arch):

    def training_step(self, batch, batch_idx,optimizers):
        (sup_x,sup_y), ((unsup_x1,unsup_x2),unsup_y) = batch
        device=self.device
        sup_x,sup_y=sup_x.to(device),sup_y.to(device)
        
        p10,p11,p12 = self.forward(sup_x)
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
            unsup_x1,unsup_x2=unsup_x1.to(device),unsup_x2.to(device)
            with torch.no_grad():
                up10,up11,up12 = self.forward(unsup_x1)
                up20,up21,up22 = self.forward(unsup_x2)
                up10,up11,up12,up20,up21,up22=up10.detach(),up11.detach(),up12.detach(),up20.detach(),up21.detach(),up22.detach()
                
            # smse0,smse1,smse2 are mse loss of sup_x1,sup_x2, umse0,umse1,umse2 are mse loss of unsup_x1,unsup_x2
            
            umse0,umse1,umse2=F.mse_loss(up10,up20),F.mse_loss(up11,up21),F.mse_loss(up12,up22)
            
            # current model dominated by other 2 models 4 mean hx 
            T=self.conf.arch.temperature
            uh10,uh11,uh12,uh20,uh21,uh22 =F.softmax(up10/T,dim=1),F.softmax(up11/T,dim=1),F.softmax(up12/T,dim=1),F.softmax(up20/T,dim=1),F.softmax(up21/T,dim=1),F.softmax(up22/T,dim=1)
            uprob0,ulabel0=torch.max((uh11+uh21+uh12+uh22)/4,dim=1)
            uprob1,ulabel1=torch.max((uh10+uh20+uh12+uh22)/4,dim=1)
            uprob2,ulabel2=torch.max((uh10+uh20+uh11+uh21)/4,dim=1)
            
            # set prob to 0 if prob < h/cnt by class
            for idx2,uprob,ulabel in zip(range(3),[uprob0,uprob1,uprob2],[ulabel0,ulabel1,ulabel2]):
                for cls in range(self.conf.dataset.num_classes):
                    cls_mask=ulabel.eq(cls)
                    prob_mask=uprob.lt(self.h_last[idx2][cls])
                    uprob[cls_mask*prob_mask]=0
            
            mask0 = uprob0.ge(self.conf.arch.threshold).float()
            mask1 = uprob1.ge(self.conf.arch.threshold).float()
            mask2 = uprob2.ge(self.conf.arch.threshold).float()
            
            # 随便选ww第一个
            ce0=torch.mean(F.cross_entropy(up10,ulabel0,reduction='none')*mask0)
            ce1=torch.mean(F.cross_entropy(up11,ulabel1,reduction='none')*mask1)
            ce2=torch.mean(F.cross_entropy(up12,ulabel2,reduction='none')*mask2)
            r=rampup(self.current_epoch)
            l0+=(umse0+ce0)*r
            l1+=(umse1+ce1)*r
            l2+=(umse2+ce2)*r
        
        loss=[l0,l1,l2]
        for i,l in enumerate(loss):
            self.log({f'train{i}_loss':l.item()})
            optimizers[i].zero_grad()
            l.backward()
            optimizers[i].step()
                
        
        return {'loss':loss,'pred':[p10,p11,p12],'y':sup_y}
    
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

def compute_kl_loss( p, q):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# pi + R-Drop
# CE + KL / MSE + KL
# use sum of p/q to compute MSE loss and Hx
class Arch(_Arch):
        
    def forward(self,x):
        m1,m2,m3=self.models
        x1,x2,x3=x[:,0,:],x[:,1,:],x[:,2,:]
        return m1(x1),m2(x2),m3(x3)
    
    
    def training_step(self, batch, batch_idx,optimizers):
        (sup_x,sup_y), ((unsup_x1,unsup_x2),unsup_y) = batch
        device=self.device
        sup_x,sup_y=sup_x.to(device),sup_y.to(device)
        
        p10,p11,p12 = self.forward(sup_x)
        q10,q11,q12 = self.forward(sup_x)
        # l0,l1,l2 are the loss of ith model, currently CE+KL
        l0=(F.cross_entropy(p10, sup_y)+F.cross_entropy(q10, sup_y))/2+compute_kl_loss(p10,q10)
        l1=(F.cross_entropy(p11, sup_y)+F.cross_entropy(q11, sup_y))/2+compute_kl_loss(p11,q11)
        l2=(F.cross_entropy(p12, sup_y)+F.cross_entropy(q12, sup_y))/2+compute_kl_loss(p12,q12)
        
        h0,h1,h2 = (F.softmax(p10, dim=1)+F.softmax(q10,dim=1))/2,(F.softmax(p11, dim=1)+F.softmax(q11, dim=1))/2,(F.softmax(p12, dim=1)+F.softmax(q12, dim=1))/2
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
                uq10,uq11,uq12 = self.forward(unsup_x1)
                up20,up21,up22 = self.forward(unsup_x2)
                uq20,uq21,uq22 = self.forward(unsup_x2)
                up10,up11,up12,up20,up21,up22=up10.detach(),up11.detach(),up12.detach(),up20.detach(),up21.detach(),up22.detach()
                uq10,uq11,uq12,uq20,uq21,uq22=uq10.detach(),uq11.detach(),uq12.detach(),uq20.detach(),uq21.detach(),uq22.detach()
            #umse0,umse1,umse2 are mse loss of unsup_x1,unsup_x2
            r=rampup(self.current_epoch)
            ukl0,ukl1,ukl2=compute_kl_loss(up10,uq10),compute_kl_loss(up11,uq11),compute_kl_loss(up12,uq12)
            umse0,umse1,umse2=F.mse_loss(up10+uq10,up20+uq20),F.mse_loss(up11+uq11,up21+uq21),F.mse_loss(up12+uq12,up22+uq22)
            
            l0+=(umse0+ukl0)*r
            l1+=(umse1+ukl1)*r
            l2+=(umse2+ukl2)*r
        
        loss=[l0,l1,l2]
        for i,l in zip(range(3),loss):
            self.log({f'train{i}_loss':l.item()})
            optimizers[i].zero_grad()
            l.backward()
            optimizers[i].step()
                
        
        return {'loss':loss,'pred':[p10,p11,p12],'y':sup_y}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pred,y=outputs['pred'],outputs['y']
        for i in range(3):
            pi,mi=pred[i],self.train_metrics[i]
            metrics=mi(pi,y)
            self.log(metrics)
        
    
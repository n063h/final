from datetime import datetime
import numpy as np
import torch
from  torch import nn
import torch.nn.functional as F
import wandb
from arch.tri import Arch as _Arch
from utils.base import Config, Timer
from utils.ema import EMA
from utils.metrics import get_multiclass_acc_metrics
from utils.ramps import exp_rampup


rampup=exp_rampup(40)

def one_hot(targets, nClass):
    logits = torch.zeros(targets.size(0), nClass).to(targets.device)
    return logits.scatter_(1,targets.unsqueeze(1),1)

def mixup_one_target(x, y, alpha=1.0, device='cuda', is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias: lam = max(lam, 1-lam)

    index = torch.randperm(x.size(0)).to(device)

    mixed_x = lam*x + (1-lam)*x[index, :]
    mixed_y = lam*y + (1-lam)*y[index]
    return mixed_x, mixed_y, lam

def compute_kl_loss( p, q):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def ce_loss(logit,y):
    return -torch.mean(torch.sum(y* F.log_softmax(logit,dim=-1), dim=-1))

# pi + R-Drop
# CE + KL / MSE + KL
# use all sets of p/q to compute MSE loss and Hx
class Arch(_Arch):
        
    def forward(self,x):
        m1,m2,m3=self.models
        x1,x2,x3=x[:,0,:],x[:,1,:],x[:,2,:]
        return m1(x1),m2(x2),m3(x3)
    
    
    def training_step(self, batch, batch_idx,optimizers):
        (sup_x,sup_y), ((un_x1,un_x2),_) = batch
        device=self.device
        sup_x,sup_y=sup_x.to(device),sup_y.to(device)
        un_x1,un_x2=un_x1.to(device),un_x2.to(device)
        optimizer=optimizers[0]
        T=self.conf.arch.temperature
        with torch.no_grad():
            unsup_x=torch.cat((un_x1,un_x2)) # unsup_x.shape=(2*unsup_size, 3, 228)
            un_logit0,un_logit1,un_logit2=self.forward(unsup_x) # logit_i means aixs=i
            un_logit01,un_logit02=un_logit0.chunk(2) # logit_i means aixs=i, sup_idx=j
            un_logit11,un_logit12=un_logit1.chunk(2) # logit_i means aixs=i, sup_idx=j
            un_logit21,un_logit22=un_logit2.chunk(2) # logit_i means aixs=i, sup_idx=j
            un_logit0=(torch.softmax(un_logit01,dim=-1) + torch.softmax(un_logit02,dim=-1))/2 # average logit of 2 output of 0th model/axis
            un_logit1=(torch.softmax(un_logit11,dim=-1) + torch.softmax(un_logit12,dim=-1))/2 # average logit of 2 output of 1st model/axis
            un_logit2=(torch.softmax(un_logit21,dim=-1) + torch.softmax(un_logit22,dim=-1))/2 # average logit of 2 output of 2nd model/axis
            un_logit0,un_logit1,un_logit2 = un_logit0**(1/T),un_logit1**(1/T),un_logit2**(1/T)
            un_y0,un_y1,un_y2= un_logit0/un_logit0.sum(dim=-1, keepdim=True).detach(),un_logit1/un_logit1.sum(dim=-1, keepdim=True).detach(),un_logit2/un_logit2.sum(dim=-1, keepdim=True).detach()
            sup_y_onehot=one_hot(sup_y,un_logit0.shape[-1])
        
        input_x=torch.cat((sup_x,un_x1,un_x2)) # (sup_size+2*unsup_size, 3, 228) -> (sup_size+2*unsup_size, 3, 10)
        un_y=torch.stack((un_y0,un_y1,un_y2),dim=1) # (unsup_size, 3, 10)
        sup_y_onehot=torch.stack((sup_y_onehot,sup_y_onehot,sup_y_onehot),dim=1) # (unsup_size, 3, 10)
        input_y=torch.cat((sup_y_onehot.expand(sup_y_onehot.shape[0],3,sup_y_onehot.shape[-1]),un_y,un_y)) # (sup_size+2*unsup_size, 3, 10)
        mixed_x, mixed_y, lam = mixup_one_target(input_x, input_y,
                                            self.conf.arch.alpha,
                                            self.device,
                                            is_bias=True)
        
        mixed_logit00,mixed_logit01,mixed_logit02 = self.forward(mixed_x) # ith model/axis pred, (size,228)
        mixed_logit10,mixed_logit11,mixed_logit12 = self.forward(mixed_x) # ith model/axis pred, (size,228)
        kl_loss0= compute_kl_loss(mixed_logit00, mixed_logit10)
        kl_loss1= compute_kl_loss(mixed_logit01, mixed_logit11)
        kl_loss2= compute_kl_loss(mixed_logit02, mixed_logit12)
        
        sup_num=sup_x.shape[0]
        l0=ce_loss(mixed_logit00[:sup_num],mixed_y[:sup_num,0,:])+ce_loss(mixed_logit10[:sup_num],mixed_y[:sup_num,0,:])/2+kl_loss0*4
        l1=ce_loss(mixed_logit01[:sup_num],mixed_y[:sup_num,1,:])+ce_loss(mixed_logit11[:sup_num],mixed_y[:sup_num,1,:])/2+kl_loss1*4
        l2=ce_loss(mixed_logit02[:sup_num],mixed_y[:sup_num,2,:])+ce_loss(mixed_logit12[:sup_num],mixed_y[:sup_num,2,:])/2+kl_loss2*4
        
        mixed_prob00=torch.softmax(mixed_logit00,dim=-1)
        mixed_prob01=torch.softmax(mixed_logit01,dim=-1)
        mixed_prob02=torch.softmax(mixed_logit02,dim=-1)
        mixed_prob10=torch.softmax(mixed_logit10,dim=-1)
        mixed_prob11=torch.softmax(mixed_logit11,dim=-1)
        mixed_prob12=torch.softmax(mixed_logit12,dim=-1)
        if self.conf.semi:
            u0= (F.mse_loss(mixed_prob00[sup_num:], mixed_y[sup_num:,0,:])+F.mse_loss(mixed_prob10[sup_num:], mixed_y[sup_num:,0,:]))/2
            u1= (F.mse_loss(mixed_prob01[sup_num:], mixed_y[sup_num:,1,:])+F.mse_loss(mixed_prob11[sup_num:], mixed_y[sup_num:,1,:]))/2
            u2= (F.mse_loss(mixed_prob02[sup_num:], mixed_y[sup_num:,2,:])+F.mse_loss(mixed_prob12[sup_num:], mixed_y[sup_num:,2,:]))/2
            l0+=rampup(self.current_epoch)*u0*self.conf.arch.usp_weight
            l1+=rampup(self.current_epoch)*u1*self.conf.arch.usp_weight
            l2+=rampup(self.current_epoch)*u2*self.conf.arch.usp_weight
            
        loss=[l0,l1,l2]
        optimizers[2].zero_grad()
        l2.backward()
        optimizers[2].step()
        # for i,l in zip(range(3),loss):
        #     self.log({f'train{i}_loss':l.item()})
        #     optimizers[i].zero_grad()
        #     l.backward()
        #     optimizers[i].step()
                
        
        return {'loss':loss,'pred':torch.stack((mixed_logit00,mixed_logit01,mixed_logit02),dim=1),'y':mixed_y.max(dim=-1)[1]}
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pred,y=outputs['pred'],outputs['y']
        for i in range(3):
            pi,mi=pred[:,i,:],self.train_metrics[i]
            metrics=mi(pi,y[:,i])
            self.log(metrics)
                
        
    
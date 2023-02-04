import numpy as np
import torch
import wandb

from utils.ramps import exp_rampup
from .base import BaseModel
import torch.nn.functional as F

def compute_kl_loss( p, q):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def consistency_loss(p,q):
    return F.mse_loss(F.softmax(p,1,dim=-1), F.softmax(q,1,dim=-1))

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

rampup=exp_rampup(40)


# mixmatch + R-Drop
class Arch(BaseModel):
    def training_step(self, batch, batch_idx,optimizers):
        (sup_x,sup_y), ((un_x1,un_x2),_) = batch
        device=self.device
        sup_x,sup_y=sup_x.to(device),sup_y.to(device)
        un_x1,un_x2=un_x1.to(device),un_x2.to(device)
        optimizer=optimizers[0]
        T=self.conf.arch.temperature
        with torch.no_grad():
            unsup_x=torch.cat((un_x1,un_x2))
            un_logit1,un_logit2=self.model(unsup_x).chunk(2)
            un_logit=(torch.softmax(un_logit1,dim=-1) + torch.softmax(un_logit2,dim=-1))/2
            un_logit = un_logit**(1/T)
            un_y= un_logit/un_logit.sum(dim=-1, keepdim=True).detach()
            sup_y_onehot=one_hot(sup_y,un_logit.shape[-1])
        input_x = torch.cat([sup_x, un_x1, un_x2])
        input_y = torch.cat([sup_y_onehot, un_y, un_y])
        ## forward
        mixed_x, mixed_y, lam = mixup_one_target(input_x, input_y,
                                                    self.conf.arch.alpha,
                                                    self.device,
                                                    is_bias=True)
        
        mixed_outputs1 = self.model(mixed_x)
        mixed_outputs2 = self.model(mixed_x)
        sup_num=sup_x.shape[0]
        
        
        loss_ce1=-torch.mean(torch.sum(mixed_y[:sup_num]* F.log_softmax(mixed_outputs1[:sup_num],dim=-1), dim=-1))
        loss_ce2=-torch.mean(torch.sum(mixed_y[:sup_num]* F.log_softmax(mixed_outputs2[:sup_num],dim=-1), dim=-1))
        kl_loss= compute_kl_loss(mixed_outputs1, mixed_outputs2)
        loss=(loss_ce1+loss_ce2)/2+kl_loss*4
        mixed_prob1=torch.softmax(mixed_outputs1,dim=-1)
        mixed_prob2=torch.softmax(mixed_outputs2,dim=-1)
        
        if self.conf.semi:
            unsup_loss = (F.mse_loss(mixed_prob1[sup_num:], mixed_y[sup_num:])+F.mse_loss(mixed_prob2[sup_num:], mixed_y[sup_num:]))/2
            loss+=rampup(self.current_epoch)*unsup_loss*self.conf.arch.usp_weight
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss':loss.item(),'pred':mixed_outputs1,'y':mixed_y.max(dim=-1)[1]}
    
    def pred_to_onehot(self,pred):
        return F.one_hot(pred.argmax(dim=1),num_classes=self.conf.dataset.num_classes)
import torch
import wandb

from utils.ramps import exp_rampup
from .base import BaseModel
import torch.nn.functional as F

def consistency_loss(p,q):
    return F.mse_loss(F.softmax(p,1,dim=-1), F.softmax(q,1,dim=-1))

rampup=exp_rampup(40)

class Arch(BaseModel):
    def training_step(self, batch, batch_idx,optimizers):
        (sup_x,sup_y), ((un_x1,un_x2),un_y) = batch
        device=self.device
        sup_x,sup_y=sup_x.to(device),sup_y.to(device)
        un_x1,un_x2=un_x1.to(device),un_x2.to(device)
        optimizer=optimizers[0]
        
        unsup_x=torch.cat((un_x1,un_x2))
        
        if self.conf.semi:
            
            with torch.no_grad():
                un_pred1 = self.model(un_x1).detach()
                un_pred2 = self.model(un_x2).detach()
            unsup_loss=consistency_loss(un_pred1,un_pred2)
            self.log({'unsup_loss':unsup_loss.item()})
            loss+=rampup(self.current_epoch)*unsup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss':loss.item(),'pred':sup_pred,'y':sup_y}
    
    def pred_to_onehot(self,pred):
        return F.one_hot(pred.argmax(dim=1),num_classes=self.conf.dataset.num_classes)
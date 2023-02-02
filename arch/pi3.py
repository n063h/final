import torch
import wandb

from utils.ramps import exp_rampup
from .base import BaseModel
import torch.nn.functional as F


rampup=exp_rampup(40)

class Arch(BaseModel):
    def training_step(self, batch, batch_idx,optimizers):
        (sup_x,sup_y), ((un_x1,un_x2),un_y) = batch
        device=self.device
        sup_x,sup_y=sup_x.to(device),sup_y.to(device)
        optimizer=optimizers[0]
        sup_pred = self.model(sup_x)
        loss = F.cross_entropy(sup_pred, sup_y)
        self.log({"sup_loss":loss.item()})
        
        if self.conf.semi:
            un_x1,un_x2=un_x1.to(device),un_x2.to(device)
            input=torch.cat((un_x1,un_x2))
            un_pred1,un_pred2 = self.model(input).chunk(2,dim=0)
            unsup_loss=F.mse_loss(un_pred1,un_pred2)
            self.log({'unsup_loss':unsup_loss.item()})
            loss+=rampup(self.current_epoch)*unsup_loss.detach()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss':loss.item(),'pred':sup_pred,'y':sup_y}
    
    def pred_to_onehot(self,pred):
        return F.one_hot(pred.argmax(dim=1),num_classes=self.conf.dataset.num_classes)
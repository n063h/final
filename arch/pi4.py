import torch
import wandb

from utils.ramps import exp_rampup
from .base import BaseModel
import torch.nn.functional as F


rampup=exp_rampup(40)
def compute_kl_loss( p, q):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss
class Arch(BaseModel):
    # pi + R-Drop(mse of sum of p/q)
    def training_step(self, batch, batch_idx,optimizers):
        (sup_x,sup_y), ((un_x1,un_x2),un_y) = batch
        device=self.device
        sup_x,sup_y=sup_x.to(device),sup_y.to(device)
        optimizer=optimizers[0]
        sup_pred1 = self.model(sup_x)
        sup_pred2 = self.model(sup_x)
        loss = (F.cross_entropy(sup_pred1, sup_y)+F.cross_entropy(sup_pred2, sup_y))/2
        self.log({"sup_loss":loss.item()})
        
        if self.conf.semi:
            un_x1,un_x2=un_x1.to(device),un_x2.to(device)
            with torch.no_grad():
                up1 = self.model(un_x1).detach()
                uq1 = self.model(un_x1).detach()
                up2 = self.model(un_x2).detach()
                uq2 = self.model(un_x2).detach()
            kl_loss=(compute_kl_loss(up1,uq1)+compute_kl_loss(up2,uq2))/2
            unsup_loss=kl_loss+F.mse_loss(up1+uq1,up2+up2)
            self.log({'unsup_loss':unsup_loss.item()})
            loss+=rampup(self.current_epoch)*unsup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss':loss.item(),'pred':sup_pred1,'y':sup_y}
    
    def pred_to_onehot(self,pred):
        return F.one_hot(pred.argmax(dim=1),num_classes=self.conf.dataset.num_classes)
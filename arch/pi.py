import torch

from utils.ramps import exp_rampup
from .base import BaseModel
import torch.nn.functional as F


rampup=exp_rampup(30)

class Arch(BaseModel):
    def training_step(self, batch, batch_idx,optimizers):
        ((sup_x1,sup_x2),sup_y), ((un_x1,un_x2),un_y) = batch
        optimizer=optimizers
        sup_pred1 = self.model(sup_x1)
        loss = F.cross_entropy(sup_pred1, sup_y)
        self.log("sup_loss", loss)
        
        if self.conf.semi:
            with torch.no_grad():
                sup_pred2 = self.model(sup_x2).detach()
                un_pred1 = self.model(un_x1).detach()
                un_pred2 = self.model(un_x2).detach()
            unsup_loss=F.mse_loss(sup_pred1,sup_pred2)+F.mse_loss(un_pred1,un_pred2)
            self.log_dict({'unsup_loss':unsup_loss},prog_bar=True,on_step=True)
            loss+=rampup(self.current_epoch)*unsup_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss':loss,'pred':sup_pred1,'y':sup_y}
    
    def pred_to_onehot(self,pred):
        return F.one_hot(pred.argmax(dim=1),num_classes=self.conf.num_classes)
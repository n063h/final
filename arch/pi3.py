import torch

from utils.ramps import exp_rampup
from .base import BaseModel
import torch.nn.functional as F


rampup=exp_rampup(30)

class Arch(BaseModel):
    def training_step(self, batch, batch_idx):
        ((sup_x1,sup_x2),sup_y), ((un_x1,un_x2),un_y) = batch
        sup_pred1 = self.model(sup_x1)
        loss = F.cross_entropy(sup_pred1, sup_y)
        self.log("sup_loss", loss.item())
        
        if self.conf.semi:
            sup_num=sup_x2.shape[0]
            input=torch.cat((sup_x2,un_x1,un_x2))
            output = self.model(input)
            sup_pred2=output[:sup_num]
            un_pred1,un_pred2=output[sup_num:].chunk(2,dim=0)
            unsup_loss=F.mse_loss(sup_pred1,sup_pred2)+F.mse_loss(un_pred1,un_pred2)
            self.log_dict({'unsup_loss':unsup_loss.item()},prog_bar=True,on_step=True)
            loss+=rampup(self.current_epoch)*unsup_loss.detach()
        return {'loss':loss.item(),'pred':sup_pred1.item(),'y':sup_y}
    
    def pred_to_onehot(self,pred):
        return F.one_hot(pred.argmax(dim=1),num_classes=self.conf.dataset.num_classes)
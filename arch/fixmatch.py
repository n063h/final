import torch
from .base import BaseModel
import torch.nn.functional as F




class Arch(BaseModel):
    def training_step(self, batch, batch_idx,optimizers):
        (sup_x,sup_y), ((un_w_x,un_s_x),un_y) = batch
        optimizer=optimizers[0]
        sup_pred = self.model(sup_x)
        loss = F.cross_entropy(sup_pred, sup_y)
        self.log("sup_loss", loss)
        
        if self.conf.semi:
            with torch.no_grad():
                u_s_pred=self.model(un_s_x).detach()
                u_w_pred=self.model(un_w_x).detach()
            u_w_prob=F.softmax(u_w_pred,dim=1)
            u_w_maxprob,u_w_label=u_w_prob.max(dim=1)
            mask = u_w_maxprob.ge(self.conf.threshold).float()
            unsup_loss = torch.mean(F.cross_entropy(u_s_pred, u_w_label, reduction='none')*mask)
            self.log_dict({'unsup_loss':unsup_loss},prog_bar=True,on_step=True)
            loss+=self.conf.usp_weight*unsup_loss
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {'loss':loss,'pred':sup_pred,'y':sup_y}
    
    def pred_to_onehot(self,pred):
        return F.one_hot(pred.argmax(dim=1),num_classes=self.conf.num_classes)
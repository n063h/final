from importlib import import_module
import hydra,wandb,torch,easydict
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from utils.device import get_pytorch_device
from utils.nir_aug import BaseTransform, DA_MagWarp, DA_Scaling,DA_Jitter, build_augs
from utils.seed import set_seed
from utils.seperate import dash_print
from torch import nn

# augs={
#         # 'jitter':[(DA_Jitter,0.05,0),(DA_Jitter,0.02,0),(DA_Jitter,0.10,0),(DA_Jitter,0.05,0.05)],
#         # 'scaling':[(DA_Scaling,0.1,1),(DA_Scaling,0.05,1),(DA_Scaling,0.2,1),(DA_Scaling,0.1,0.5)],
#         'scaling':[(DA_Scaling,0.1,1),(DA_Scaling,0.2,1)],
#         # 'magwarp':[(DA_MagWarp,0.05,0),(DA_MagWarp,0.02,0),(DA_MagWarp,0.10,0),(DA_MagWarp,0.05,0.05)],
#         'magwarp':[(DA_MagWarp,0.02,0),(DA_MagWarp,0.10,0)],
# }



@hydra.main(version_base=None, config_path="conf", config_name="config")
def init_conf(_conf):
    set_seed(_conf.seed)
    global conf_cp
    conf_cp=easydict.EasyDict(OmegaConf.to_container(_conf, resolve=True))





    
    
def sweep():
    global conf_cp
    wandb.init(
        project=conf_cp.project,
        entity=conf_cp.entity,
        name=conf_cp.name,
        notes=conf_cp.name,
        config=conf_cp
    )
    conf = easydict.EasyDict(wandb.config)
    conf.device=get_pytorch_device()
    conf.dataset.w_augs=build_augs(conf.dataset.w_augs,conf.alpha,conf.beta)
    conf.dataset.s_augs=build_augs(conf.dataset.s_augs,conf.alpha,conf.beta)
    dataset:LightningDataModule=import_module('datasets.'+conf.dataset.name).Dataset(conf)
    model:nn.Module=import_module('models.'+conf.model.name).Net(conf,conf.device)
    arch=import_module('arch.'+conf.arch.name).Arch(model=model,conf=conf,device=conf.device)
    arch.fit(dataset)
    
    
    
    
if __name__ == "__main__":
    init_conf()
    global conf_cp
    sweep_configuration = {
        'method': conf_cp.sweep_method,
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'val_acc'},
        'parameters': 
        {
            'alpha': {'max': 1, 'min': -1},
            'beta': {'max': 1, 'min': -1}
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=conf_cp.project)
    wandb.agent(sweep_id,
                project=conf_cp.project,
                entity=conf_cp.entity,
                function=sweep, 
                count=30
            )
from importlib import import_module
import hydra,wandb
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from utils.device import get_pytorch_device
from utils.seperate import dash_print
from torch import nn



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    conf_dict=OmegaConf.to_container(conf, resolve=True)
    dash_print(conf_dict)
    wandb.init(
        project=conf.project,
        entity=conf.entity,
        name=conf.name,
        notes=conf.name,
        config=conf_dict
    )
    device=get_pytorch_device()
    dataset:LightningDataModule=import_module('datasets.'+conf.dataset.name).Dataset(conf)
    model:nn.Module=import_module('models.'+conf.model.name).Net(conf,device)
    arch=import_module('arch.'+conf.arch.name).Arch(model=model,conf=conf,device=device)
    arch.fit(dataset)
    wandb.finish()
    
if __name__ == "__main__":
    main()
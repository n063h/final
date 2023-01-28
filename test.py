from importlib import import_module
import hydra,wandb,torch,easydict
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from utils.device import get_pytorch_device
from utils.nir_aug import BaseTransform, DA_MagWarp, DA_Scaling
from utils.seperate import dash_print
from torch import nn

augs={
        # 'jitter':[(DA_Jitter,0.05,0),(DA_Jitter,0.02,0),(DA_Jitter,0.10,0),(DA_Jitter,0.05,0.05)],
        # 'scaling':[(DA_Scaling,0.1,1),(DA_Scaling,0.05,1),(DA_Scaling,0.2,1),(DA_Scaling,0.1,0.5)],
        'scaling':[(DA_Scaling,0.1,1),(DA_Scaling,0.2,1)],
        # 'magwarp':[(DA_MagWarp,0.05,0),(DA_MagWarp,0.02,0),(DA_MagWarp,0.10,0),(DA_MagWarp,0.05,0.05)],
        'magwarp':[(DA_MagWarp,0.02,0),(DA_MagWarp,0.10,0)],
}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    torch.manual_seed(2)
    conf_dict=OmegaConf.to_container(conf, resolve=True)
    dash_print(conf_dict)
    wandb.init(
        project=conf.project,
        entity=conf.entity,
        name=conf.name,
        notes=conf.name,
        config=conf_dict
    )
    conf=easydict.EasyDict(conf_dict)
    conf.device=get_pytorch_device()
    conf.dataset.w_augs=[BaseTransform(*augs['magwarp'][0])]
    conf.dataset.s_augs=[BaseTransform(*augs['magwarp'][0]),BaseTransform(*augs['scaling'][0])]
    dataset:LightningDataModule=import_module('datasets.'+conf.dataset.name).Dataset(conf)
    model:nn.Module=import_module('models.'+conf.model.name).Net(conf,conf.device)
    arch=import_module('arch.'+conf.arch.name).Arch(model=model,conf=conf,device=conf.device)
    arch.fit(dataset)
    wandb.finish()
    
if __name__ == "__main__":
    main()
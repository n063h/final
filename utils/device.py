import os,torch

def get_lightning_device():
    accelerator,devices='cpu',None
    if "COLAB_TPU_ADDR" in os.environ:
        accelerator,devices='tpu',8
    elif torch.cuda.is_available():
        accelerator,devices='gpu',1
    return {'accelerator':accelerator,'devices':devices}

def get_pytorch_device():
    device='cpu'
    if torch.cuda.is_available():
        device='cuda'
    return torch.device(device)
import torch
from torch import nn
from torch.utils import data
import numpy as np
import json,random
from torch.utils.data import Dataset
import torchvision.transforms as transforms

cls50 = ['_R_P_SP','_C_L','_L','_C','_P_SP','_R_SP_N','_C_P','_P_W','_R_P','_P','_C_P_SP','_R_P_SP_N','_R','_C_SP_N','_W','_R_SP','_W_SP','_C_SP','_R_L','_W_N_A','_C_N','_P_A','_SP_N','_R_N','_W_A','_N','_P_SP_A','_P_SP_N_A','_N_A','_P_N','_V_N','_P_W_A','_P_W_N_A','_W_SP_N_A','_CA','_SP_N_A','_R_TEN','_C_A','_TEN','_R_V','_W_CA','_P_W_SP_N_A','_C_R','_P_W_SP_A','_S','_W_S','_P_N_A','_M_P','_M_SP','_V_P']
cls10=['_R_P_SP', '_C_L', '_L', '_P_SP', '_R_SP_N', '_C_P', '_P_W', '_P','_C_P_SP', '_R_P_SP_N']

cls_label={
    'cls10':cls10,
    'cls50':cls50
}

def read_line(line,cls_num):
    if not line:
        return
    try:
        data=json.loads(line)
    except Exception as e:
        return
    labels=cls_label[cls_num]
    label,d=list(data.items())[0]
    if label not in labels:
        return
    _y=labels.index(label)
    _x=[d['i'],d['r'],d['a']]
    return _x,_y
    
def extract_json(filename,cls_num):
    x,y=[],[]
    with open(filename,encoding='UTF-8') as f:
        for line in f.readlines():
            ret=read_line(line,cls_num)
            if not ret:
                continue
            x.append(ret[0])
            y.append(ret[1])
    return x,y

def save_to_npy(x,y,filename):
    np.save(filename+'_x.npy',x)
    np.save(filename+'_y.npy',y)
    
def read_npy(filename):
    x=np.load(filename+'_x.npy')
    y=np.load(filename+'_y.npy')
    return x,y

def get_random_ratio(max_ratio=0.2):
    max_percentage=max_ratio*100
    return random.randrange(max_percentage/5,max_percentage)/100

def get_random_length(wave,max_ratio=0.1):
    random_ratio=get_random_ratio(max_ratio)
    return int(random_ratio*len(wave))

def get_random_range(wave,max_ratio=0.1):
    random_length=get_random_length(wave,max_ratio)
    end_max=len(wave)-random_length
    start=torch.randint(0,end_max,[1])[0]
    return (start,start+random_length)



def random_up_down(wave,value,wave_length_range=(0,228)):
    start,end=wave_length_range
    if random.random() < 0.5:
        value=-value
    value/=100
    wave[start:end]+=value
    return wave

def gauss_white_noise(wave,means=0,std=0.1):
    noise=torch.normal(means, std, wave.shape)
    return wave+noise

def augment_pool():
    augs = [(random_up_down, 20, 5),
            (gauss_white_noise,None,None)]
    return augs


class RandAugmentNir(object):
    def __init__(self, choice_num, max_value_ratio):
        self.choice_num = choice_num
        self.max_value_ratio = max_value_ratio # control weak and strong  aug limitation
        self.augment_pool = augment_pool()

    def __call__(self, origin):
        ops = random.choices(self.augment_pool, k=self.choice_num)
        x=torch.clone(origin)
        for op, max_value, bias in ops:
            if not bias:
                 x = op(x)
            else:
                max_v=max_value*self.max_value_ratio
                value = np.random.randint(1, max_v)
                if random.random() < 0.5:
                    x = op(x, value=value+bias)
        return x

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, origin):
        return torch.Tensor(origin).float()

class NIRDataset(Dataset):
    def __init__(self, data,targets,transfoms,num_classes,axis=None):
        self.data=data
        self.targets=targets
        self.transfoms=transfoms
        self.num_classes=num_classes
        self.axis=axis
        if axis is not None:
            self.axis_data=data[:,axis]
        else:
            self.axis_data=data
            
    def __getitem__(self, index):
        x,y=self.axis_data[index],self.targets[index]
        if self.axis is not None:
            _x=self.transfoms(x)
        else:
            _x=x
        return _x,y

    def __len__(self):
        return len(self.axis_data)
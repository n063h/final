import itertools
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict,Counter
import numpy as np
from torch.utils.data import Dataset,Subset
import torchvision.transforms as transforms

NO_LABEL = -1

class DataSetWarpper(Dataset):
    """Enable dataset to output index of sample
    """
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes

    def __getitem__(self, index):
        sample, label = self.dataset[index]
        return sample, label, index

    def __len__(self):
        return len(self.dataset)

class TransformTwice:

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TransformWeakStrong:

    def __init__(self, trans1, trans2):
        self.transform1 = trans1
        self.transform2 = trans2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2

class TransformBaseWeakStrong:

    def __init__(self, trans0,trans1, trans2):
        self.transform0 = trans0
        self.transform1 = trans1
        self.transform2 = trans2

    def __call__(self, inp):
        base=self.transform0(inp)
        out1 = self.transform1(base)
        out2 = self.transform2(base)
        return out1, out2

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            secondary_batch + primary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices, is_shuffle=True):
    shuffleFunc = np.random.permutation if is_shuffle else lambda x: x
    def infinite_shuffles():
        while True:
            yield shuffleFunc(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)]*n
    return zip(*args)


def uniform_split_subset(dataset,indice,length,shuffle=False):
    full_targets,indices=np.array(dataset.targets),np.array(indice)
    sub_targets=full_targets[indices]
    unique_labels=np.unique(sub_targets)
    subsets=[]
    used_label_num=defaultdict(int)
    for ratio in length:
        # for every item in length, construct a subset
        indice=[]
        for y in unique_labels:
            indice_idxes=np.where(sub_targets==y)[0]
            targets_idxes=indices[indice_idxes]
            cur=int(ratio*len(indice_idxes))
            used=used_label_num[y]
            indice.extend(targets_idxes[used:used+cur])
            used_label_num[y]+=cur
        if shuffle: np.random.shuffle(indice)
        subsets.append(Subset(dataset, indice))
    return subsets

def uniform_split_dataset(dataset,length,shuffle=True):
    targets=np.array(dataset.targets)
    subsets,unique_labels,used_label_num=[],np.unique(targets),defaultdict(int)
    for ratio in length:
        # for every item in length, construct a subset
        indice=[]
        for y in unique_labels:
            # for every label, append ratio*len idxes into indice
            idxes=np.where(targets==y)[0]
            cur=int(ratio*len(idxes))
            used=used_label_num[y]
            indice.extend(idxes[used:used+cur])
            used_label_num[y]+=cur
        if shuffle: np.random.shuffle(indice)
        subsets.append(Subset(dataset, indice))
    return subsets

def uniform_split(dataset,length,shuffle=False):
    if isinstance(dataset,Subset):
        # if is subset, restore subset to full-dataset while keep indice stay subset
        indice=np.array(dataset.indices)
        while isinstance(dataset,Subset):
            dataset=dataset.dataset
        return uniform_split_subset(dataset,indice,length,shuffle)
    else:
        return uniform_split_dataset(dataset,length,shuffle)
    
class MyDataset(Dataset):
    def __init__(self, data,targets,transfom=None):
        self.data=data
        self.targets=targets
        self.transfom=transfom
            
    def __getitem__(self, index):
        x,y=self.data[index],self.targets[index]
        if self.transform:
            if isinstance(x, np.ndarray) and x.shape[0] == 3:
                pass
            else:
                x = self.transform(x)
        return x,y

    def __len__(self):
        return len(self.data)

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            if isinstance(x, np.ndarray) and x.shape[0] == 3:
                pass
            else:
                x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

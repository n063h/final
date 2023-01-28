CUDA_VISIBLE_DEVICES=2 python test.py dataset=cifar10  model=resnet arch=nosemi 2>&1| tee log2
CUDA_VISIBLE_DEVICES=2 python test.py dataset=cifar10ww  model=resnet arch=pi 2>&1| tee log2
CUDA_VISIBLE_DEVICES=2 python test.py dataset=cifar10ws  model=resnet arch=fixmatch 2>&1| tee log2
CUDA_VISIBLE_DEVICES=1 python test.py dataset=cls10  model=resnet1d arch=nosemi label_ratio=0.99 2>&1| tee log2

# shutdown
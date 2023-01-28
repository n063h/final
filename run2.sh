CUDA_VISIBLE_DEVICES=2 python test.py dataset=cifar10  model=resnet arch=nosemi
CUDA_VISIBLE_DEVICES=2 python test.py dataset=cifar10ww  model=resnet arch=pi
CUDA_VISIBLE_DEVICES=2 python test.py dataset=cifar10ws  model=resnet arch=fixmatch


shutdown
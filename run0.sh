CUDA_VISIBLE_DEVICES=0 python test.py dataset=3cls10  model=3resnet1d arch=tri
CUDA_VISIBLE_DEVICES=0 python test.py dataset=cls10ww  model=resnet1d arch=pi
CUDA_VISIBLE_DEVICES=0 python test.py dataset=cls10ws  model=resnet1d arch=fixmatch
CUDA_VISIBLE_DEVICES=0 python test.py dataset=cls10  model=resnet1d arch=nosemi


shutdown
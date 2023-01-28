#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py dataset=3cls10  model=3resnet1d arch=tri label_ratio=0.1 2>&1| tee log1_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=1 python train.py dataset=cls10ww  model=resnet1d arch=pi label_ratio=0.1 2>&1| tee log1_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=1 python train.py dataset=cls10ws  model=resnet1d arch=fixmatch label_ratio=0.1 2>&1| tee log1_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=1 python train.py dataset=cls10  model=resnet1d arch=nosemi label_ratio=0.1 2>&1| tee log1_$(date +%y-%m-%d-%H-%M).txt


# shutdown
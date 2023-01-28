#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10  model=3resnet1d arch=tri 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cls10ww  model=resnet1d arch=pi 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cls10ws  model=resnet1d arch=fixmatch 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cls10  model=resnet1d arch=nosemi 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt


# shutdown
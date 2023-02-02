#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar model=resnet arch=pi label_ratio=0.01 semi=False notes=nosemi_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar model=resnet arch=pi label_ratio=0.01 notes=pi_works_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar model=resnet arch=pi2 label_ratio=0.01 notes=pi2_works_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar model=resnet arch=pi3 label_ratio=0.01 notes=pi3_timecost_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar model=resnet arch=fixmatch label_ratio=0.01 notes=fixmatch_works_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cifar model=resnet arch=fixmatch2 label_ratio=0.01 notes=fixmatch2_works_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cls10 model=resnet1d arch=fixmatch label_ratio=0.01 notes=fixmatch_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=cls10 model=resnet1d arch=fixmatch2 label_ratio=0.01 notes=fixmatch2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# shutdown
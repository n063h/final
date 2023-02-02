#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=1 python train.py dataset=nir dataset.name=cls10 axis=0 model=resnet1d  arch=pi label_ratio=0.01 semi=False notes=nosemi_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=1 python train.py dataset=nir dataset.name=cls10 axis=0 model=resnet1d  arch=pi label_ratio=0.01 notes=pi_works_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=1 python train.py dataset=nir dataset.name=cls10 axis=0 model=resnet1d  arch=pi2 label_ratio=0.01 notes=pi2_works_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=1 python train.py dataset=nir dataset.name=cls10 axis=0 model=resnet1d  arch=pi3 label_ratio=0.01 notes=pi3_timecost_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=1 python train.py dataset=nir dataset.name=cls10 axis=0 model=resnet1d  arch=pi4 label_ratio=0.01 notes=pi4_works_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=1 python train.py dataset=nir dataset.name=cls10 axis=0 model=resnet1d  arch=pi5 label_ratio=0.01 notes=pi5_works_cifar10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

# shutdown
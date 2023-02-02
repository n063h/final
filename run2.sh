#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir model=resnet1d  arch=tri arch.name=tri1 label_ratio=0.01 notes=tri1_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir model=resnet1d  arch=tri arch.name=tri2 label_ratio=0.01 notes=tri2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir model=resnet1d  arch=tri arch.name=tri3 label_ratio=0.01 notes=tri3_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir model=resnet1d  arch=tri arch.name=tri4 label_ratio=0.01 notes=tri4_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir model=resnet1d  arch=tri arch.name=tri5 label_ratio=0.01 notes=tri5_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

# shutdown
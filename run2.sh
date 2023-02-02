#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir dataset.name=cls10 model=resnet1d  arch=tri label_ratio=0.01  notes=tri_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir dataset.name=cls10 model=resnet1d  arch=tri1 label_ratio=0.01 notes=tri1_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir dataset.name=cls10 model=resnet1d  arch=tri2 label_ratio=0.01 notes=tri2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir dataset.name=cls10 model=resnet1d  arch=tri3 label_ratio=0.01 notes=tri3_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir dataset.name=cls10 model=resnet1d  arch=tri4 label_ratio=0.01 notes=tri4_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=2 python train.py dataset=nir dataset.name=cls10 model=resnet1d  arch=tri5 label_ratio=0.01 notes=tri5_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

# shutdown
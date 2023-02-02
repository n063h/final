#!/usr/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.axis=0 model=resnet1d model.name=3resnet1d arch=pi arch.name=pi label_ratio=0.01 semi=False notes=nosemi_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.axis=0 model=resnet1d model.name=3resnet1d arch=pi arch.name=pi label_ratio=0.01  notes=pi_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.axis=0 model=resnet1d model.name=3resnet1d arch=pi arch.name=pi4 label_ratio=0.01 notes=pi4_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.axis=0 model=resnet1d model.name=3resnet1d arch=pi arch.name=pi5 label_ratio=0.01 notes=pi5_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.axis=0 model=resnet1d model.name=3resnet1d arch=tri arch.name=tri4 label_ratio=0.01 notes=tri4_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.axis=0 model=resnet1d model.name=3resnet1d arch=tri arch.name=tri5 label_ratio=0.01 notes=tri5_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# shutdown
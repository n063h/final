#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.name=cls10 model=2resnet1d axis=0 arch=pi label_ratio=0.01 semi=False notes=nosemi_2r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.name=cls10 model=2resnet1d axis=0 arch=pi label_ratio=0.01  notes=pi_2r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.name=cls10 model=2resnet1d axis=0 arch=pi4 label_ratio=0.01 notes=pi4_2r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.name=cls10 model=2resnet1d axis=0 arch=pi5 label_ratio=0.01 notes=pi5_2r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.name=cls10 model=3resnet1d axis=0 arch=pi label_ratio=0.01 semi=False notes=nosemi_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.name=cls10 model=3resnet1d axis=0 arch=pi label_ratio=0.01  notes=pi_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.name=cls10 model=3resnet1d axis=0 arch=pi4 label_ratio=0.01 notes=pi4_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=nir dataset.name=cls10 model=3resnet1d axis=0 arch=pi5 label_ratio=0.01 notes=pi5_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

# shutdown
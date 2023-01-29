#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10  model=3resnet1d arch=tri notes=3same_data_model 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10ww  model=3resnet1d arch=tri1 notes=3same_data_model_pi 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10ww  model=3resnet1d arch=tri2 notes=3same_data_model_pi_fm 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10ww  model=3resnet1d arch=tri1 dataset.indep=True notes=3same_data_model_pi_indep 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10  model=3resnet1d arch=tri label_ratio=0.1 notes=3same_data_model 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10ww  model=3resnet1d arch=tri1 label_ratio=0.1 notes=3same_data_model_pi 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10ww  model=3resnet1d arch=tri2 label_ratio=0.1 notes=3same_data_model_pi_fm 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py dataset=3cls10ww  model=3resnet1d arch=tri1 dataset.indep=True label_ratio=0.1 notes=3same_data_model_pi_indep 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

# shutdown
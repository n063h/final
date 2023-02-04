#!/usr/bin/bash
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir dataset.axis=0 model=resnet1d arch=pi arch.name=pi label_ratio=0.01 semi=False notes=nosemi_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py max_epochs=100 dataset=nir dataset.axis=0 model=resnet1d model.name=3resnet1d arch=mixmatch arch.name=mixmatch2 label_ratio=0.01  notes=mixmatch2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d arch=tri arch.name=trimix label_ratio=0.1 notes=trimix_0.1_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d arch=tri arch.name=trimix2 label_ratio=0.01 notes=trimix2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d arch=tri arch.name=trimix3 label_ratio=0.01 notes=trimix3_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d arch=tri arch.name=trimix5 label_ratio=0.01 notes=trimix5_only2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d model.name=3resnet1d arch=tri arch.name=trimix label_ratio=0.01 notes=trimix_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# shutdown
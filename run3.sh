#!/usr/bin/bash
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir dataset.axis=0 model=resnet1d arch=pi arch.name=pi label_ratio=0.01 semi=False notes=nosemi_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py max_epochs=100 dataset=nir dataset.axis=0 model=resnet1d model.name=3resnet1d arch=mixmatch arch.name=mixmatch2 label_ratio=0.01  notes=mixmatch2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d arch=tri arch.name=trimix label_ratio=0.1 notes=trimix_0.1_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d arch=tri arch.name=trimix2 label_ratio=0.01 notes=trimix2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d arch=tri arch.name=trimix3 label_ratio=0.01 notes=trimix3_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d arch=tri arch.name=trimix5 label_ratio=0.01 notes=trimix5_only2_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d model.name=3resnet1d arch=tri arch.name=trimix label_ratio=0.01 notes=trimix_3r_works_cls10 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
# shutdown


CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d model.name=3resnet1d arch=tri label_ratio=0.01 notes=no_aug_0.01 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d model.name=3resnet1d arch=tri label_ratio=0.1 notes=no_aug_0.1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=nir model=resnet1d model.name=3resnet1d arch=tri label_ratio=1 notes=no_aug_1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

# nosemi_axis1_scaling
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_scaling dataset.axis=1 alpha=0.367 beta=0.891 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=0.01 notes=nosemi_axis1_scaling_0.01 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_scaling dataset.axis=1 alpha=0.367 beta=0.891 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=0.1 notes=nosemi_axis1_scaling_0.1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_scaling dataset.axis=1 alpha=0.367 beta=0.891 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=1 notes=nosemi_axis1_scaling_1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_scaling dataset.axis=1 alpha=0.367 beta=0.891 model=resnet1d model.name=3resnet1d arch=mixmatch label_ratio=0.01 notes=mixmatch_axis1_scaling_0.01 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_scaling dataset.axis=1 alpha=0.367 beta=0.891 model=resnet1d model.name=3resnet1d arch=mixmatch label_ratio=0.1 notes=mixmatch_axis1_scaling_0.1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

# nosemi_axis1_magwarp
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=1 alpha=0.049 beta=1.539 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=0.01 notes=nosemi_axis1_magwarp_0.01 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=1 alpha=0.049 beta=1.539 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=0.1 notes=nosemi_axis1_magwarp_0.1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=1 alpha=0.049 beta=1.539 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=1 notes=nosemi_axis1_magwarp_1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=1 alpha=0.049 beta=1.539 model=resnet1d model.name=3resnet1d arch=mixmatch label_ratio=0.01 notes=mixmatch_axis1_magwarp_0.01 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=1 alpha=0.049 beta=1.539 model=resnet1d model.name=3resnet1d arch=mixmatch label_ratio=0.1 notes=mixmatch_axis1_magwarp_0.1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

# nosemi_axis2_magwarp
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=2 alpha=0.211 beta=0.835 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=0.01 notes=nosemi_axis2_magwarp_0.01 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=2 alpha=0.211 beta=0.835 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=0.1 notes=nosemi_axis2_magwarp_0.1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=2 alpha=0.211 beta=0.835 model=resnet1d model.name=3resnet1d arch=nosemi label_ratio=1 notes=nosemi_axis2_magwarp_1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=2 alpha=0.211 beta=0.835 model=resnet1d model.name=3resnet1d arch=mixmatch label_ratio=0.01 notes=mixmatch_axis2_magwarp_0.01 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt
CUDA_VISIBLE_DEVICES=0 python train.py  dataset=cls10_magwarp dataset.axis=2 alpha=0.211 beta=0.835 model=resnet1d model.name=3resnet1d arch=mixmatch label_ratio=0.1 notes=mixmatch_axis2_magwarp_0.1 2>&1| tee log0_$(date +%y-%m-%d-%H-%M).txt

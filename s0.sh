#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=0 python sweep.py dataset=cls10_jitter dataset.axis=1  model=resnet1d label_ratio=1 arch=pi semi=False 2>&1| tee sweep$(date +%y-%m-%d-%H-%M).log
# CUDA_VISIBLE_DEVICES=0 python sweep.py dataset=cls10_jitter dataset.axis=1  model=resnet1d label_ratio=0.1 arch=mixmatch 2>&1| tee sweep$(date +%y-%m-%d-%H-%M).log

CUDA_VISIBLE_DEVICES=0 python sweep.py dataset=cls10_magwarp dataset.axis=1  model=resnet1d label_ratio=1 arch=pi semi=False 2>&1| tee sweep$(date +%y-%m-%d-%H-%M).log
# CUDA_VISIBLE_DEVICES=0 python sweep.py dataset=cls10_magwarp dataset.axis=1  model=resnet1d label_ratio=0.1 arch=mixmatch 2>&1| tee sweep$(date +%y-%m-%d-%H-%M).log



# shutdown
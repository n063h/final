#!/usr/bin/bash

CUDA_VISIBLE_DEVICES=2 python sweep.py dataset=cls10_scaling  model=resnet1d label_ratio=0.99 arch=pi semi=False 2>&1| tee sweep$(date +%y-%m-%d-%H-%M).log
CUDA_VISIBLE_DEVICES=2 python sweep.py dataset=cls10_scaling  model=resnet1d label_ratio=0.99 arch=pi 2>&1| tee sweep$(date +%y-%m-%d-%H-%M).log

# shutdown
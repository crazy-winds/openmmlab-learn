#!/bin/bash
module load anaconda/2021.05
module load cuda/11.1
module load gcc/7.3

source activate mmlab
export PYTHONUNBUFFERED=1

python  train.py \
        config.py \
        --work-dir work_dir
#!/bin/bash
module load anaconda/2022.10
module load cuda/11.7
# module load gcc/7.3

source activate mmlab
export PYTHONUNBUFFERED=1

python  train.py \
        config.py \
        --work-dir work_dir
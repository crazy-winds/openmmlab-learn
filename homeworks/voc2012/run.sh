#!/bin/bash
module load anaconda/2022.10
module load cuda/11.7
# module load gcc/7.3

source activate mmlab
export PYTHONUNBUFFERED=1
export https_proxy=http://192.168.1.21:8888
export https_proxy=http://192.168.1.21:8888
export ftp_proxy=http://192.168.1.21:8888

python  train.py \
        config.py \
        --work-dir work_dir
#!/bin/sh

SRC_PATH_HPDLF=/home/diz/code/hpdlf
SRC_PATH_INN=/home/diz/code/freia_keras

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate tarantella
export PYTHONPATH=${SRC_PATH_HPDLF}/build/:${SRC_PATH_HPDLF}/src/:${SRC_PATH_INN}

python ${SRC_PATH_INN}/test_training.py

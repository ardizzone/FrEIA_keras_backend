#!/bin/sh

SRC_PATH_INN=/home/diz/code/freia_keras
OUTPUT_PATH=${SRC_PATH_INN}/output
export DATASET_DIR=/

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate tarantella
export PYTHONPATH=${SRC_PATH_INN}
python ${SRC_PATH_INN}/main.py --no-hpdlf train ${OUTPUT_PATH}/test_run

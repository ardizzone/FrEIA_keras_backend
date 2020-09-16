#!/bin/sh

SRC_PATH_HPDLF=/home/ardizzone/hpdlf_project/hpdlf
SRC_PATH_INN=/home/ardizzone/hpdlf_project/FrEIA_keras_backend

export DATASET_DIR=/home/ardizzone

. /opt/anaconda3/etc/profile.d/conda.sh

conda activate tarantella
export PYTHONPATH=${SRC_PATH_HPDLF}/build/:${SRC_PATH_HPDLF}/src/

cd ${SRC_PATH_INN}
python main.py train output/styx_test

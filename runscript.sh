#!/bin/sh

SRC_PATH_HPDLF=/home/diz/code/hpdlf
SRC_PATH_INN=/home/diz/code/freia_keras

export DATASET_DIR=/home/diz

. /opt/miniconda3/etc/profile.d/conda.sh

conda activate tarantella
export PYTHONPATH=${SRC_PATH_HPDLF}/build/:${SRC_PATH_HPDLF}/src/

cd ${SRC_PATH_INN}
python main.py train output/test_run

# with 2.1.0-gpu anaconda build h0d30ee6_0
# _ZN10tensorflow12OpDefBuilder4AttrESs

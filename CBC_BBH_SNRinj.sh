#!/usr/bin/bash

export OPENBLAS_NUM_THREAD = 1
source /opt/exp_software/virgo/jharms/.bashrc && conda activate ligo
unset LD_LIBRARY_PATH
unset PKG_CONFIG_PATH
unset PYTHONPATH
python CBC_BBH_SNRinj.py $1

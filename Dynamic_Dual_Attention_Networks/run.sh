#!/bin/bash

FPATH=$(pwd)

# cite: 16 strategies; pub: 8 strategies
STRATEGY='cite'
NSTRATEGY=16

# from Year 2000 to Year 2019
START_YEAR=2018
END_YEAR=2019

C_BATCHSIZE=12000
A_BATCHSIZE=6000
MAX_NROUND=300
MIN_NROUND=100

mkdir ${FPATH}/run # the directory to store training logs
mkdir ${FPATH}/${STRATEGY}_result # the directory to store training outputs

python3 ${FPATH}/Model/MultiHeteroAtt.py --path ${FPATH} --start_year ${START_YEAR} --end_year ${END_YEAR} --c_batchsize ${C_BATCHSIZE} --a_batchsize ${A_BATCHSIZE} --nstrategy ${NSTRATEGY} --strategy ${STRATEGY} --max_nround ${MAX_NROUND} --min_nround ${MIN_NROUND}

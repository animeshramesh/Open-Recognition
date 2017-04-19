#!/bin/bash
input='../GT_Pugh'
log='regressor_logs_L2.txt'
save_file='../models/regressor_L2.h5'

nohup python regressor.py -v -l mean_squared_error -i ${input} -s ${save_file} > ${log} &

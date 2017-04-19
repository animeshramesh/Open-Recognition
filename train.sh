#!/bin/bash
model_type=${1:-DEFAULT_MODEL}

nohup python regressor.py -v \
    --save_file ../clean_models/${model_type}.h5 \
    --model_type ${model_type} \
    &> ../clean_logs/${model_type}.log &


#!/bin/bash
output='../validation_data'
mkdir -p ${output}

log=${output}/log.txt
nohup python main.py -m -v -s 10000 -o ${output} > ${log} & 
#python main.py -m -v -s 25000 -o ${output}

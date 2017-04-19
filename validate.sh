#!/bin/bash
model_type=${1:-DEFAULT_MODEL}

python test_svm_validity.py -v \
    --base_num 10 \
    --input_model ../clean_models/${model_type}.h5 \
    --features_path /media/storage/capstone/data/ILSVRC2013/vgg16/ILSVRC2013_validate \
    --input_dir /media/storage/capstone/data/ILSVRC2013/svm_triples/validate \
    

    

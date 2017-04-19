#!/bin/bash

python main.py -v \
        --chunk 2000 \
        --samples_num 6000 \
        --input_dir /media/storage/capstone/data/ILSVRC2013/vgg16/ILSVRC2013_validate \
        --output /media/storage/capstone/data/ILSVRC2013/svm_triples/validate

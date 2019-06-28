#!/bin/bash

python test_qa.py \
-experiment Debug \
-test_dataset CASP11Stage1_SCWRL \
-load_epoch 98 \
-mult 10

python test_qa.py \
-experiment Debug \
-test_dataset CASP11Stage2_SCWRL \
-load_epoch 98 \
-mult 10

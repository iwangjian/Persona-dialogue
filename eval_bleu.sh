#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

python3 eval_bleu_v2.py data/test_data_random.json data/test_data_biased.json


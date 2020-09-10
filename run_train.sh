#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# set python path according to your actual environment
pythonpath='python3'

# train model
data_dir=./data/
save_dir=./models/
vocab_file=vocab.pt

${pythonpath} ./run_model.py --data_dir=${data_dir} --vocab_file=${vocab_file} --save_dir=${save_dir}

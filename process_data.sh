#!/bin/bash

# set python path according to your actual environment
pythonpath='python3'

# set data path
corpus_file=./data/dialogues_train.json
train_file=./data/train.json
valid_file=./data/valid.json


${pythonpath} ./tools/convert_session_to_sample.py ${corpus_file} ${train_file} ${valid_file}

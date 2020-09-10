#!/bin/bash

# set gpu id to use
export CUDA_VISIBLE_DEVICES=0

# set python path according to your actual environment
pythonpath='python3'

#test_file=./data/test.json
#test_file=./data/test_data_biased.json
test_file=./data/test_data_random.json

output_dir=./output
ckpt_prefix=best.model
mkdir -p ${output_dir}/${ckpt_prefix}

${pythonpath} ./run_model.py --test --test_file=${test_file} --ckpt=models-v2/${ckpt_prefix} --gen_file=${output_dir}/${ckpt_prefix}/output.txt


${pythonpath} ./tools/convert_result_for_eval.py ${test_file} ${output_dir}/${ckpt_prefix}/output.txt ${output_dir}/${ckpt_prefix}/output.eval.txt
${pythonpath} ./tools/eval.py ${output_dir}/${ckpt_prefix}/output.eval.txt ${output_dir}/${ckpt_prefix}/result.log

#!/bin/bash
dataset=$1
repetitions=5
vocab_size=1000

# slm
nohup python run_mlm.py --method counts --repetitions $repetitions  --vocab_size $vocab_size --dataset_name $dataset --report_type Radiology > "$dataset".slm.log &

# gru
nohup python run_mlm.py --method gru --repetitions $repetitions  --vocab_size $vocab_size --dataset_name $dataset --report_type Radiology > "$dataset".gru.log &

# lstm
nohup python run_mlm.py --method lstm --repetitions $repetitions  --vocab_size $vocab_size --dataset_name $dataset --report_type Radiology > "$dataset".lstm.log &

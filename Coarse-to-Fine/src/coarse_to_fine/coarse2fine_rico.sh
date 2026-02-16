#!/bin/bash

# Script for Single-Node multi-GPU training

MODE=$1
DATA_DIR=$2
OUT_DIR=$3
TRAINER=$4
NUM_GPU=$5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TASK_DIR="${SCRIPT_DIR}"
PROJECT_SRC_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $TRAINER = "deepspeed" ]]
then
    # DeepSpeed
    COMMAND="deepspeed --master_port 60001"
elif [[ $TRAINER = "ddp" ]]
then
    # Distributed Data Parallel
    COMMAND="python -m torch.distributed.launch --nproc_per_node=$NUM_GPU --master_port 6870"
else
    # Data Parallel
    TRAINER="basic"
    COMMAND="python"
fi

echo $COMMAND

PYTHONPATH="${PROJECT_SRC_DIR}" $COMMAND "${TASK_DIR}/main.py" --${MODE} --dataset rico \
--max_num_elements 20 \
--num_labels 25 \
--data_dir ${DATA_DIR} \
--out_dir ${OUT_DIR} \
--epoch 300 \
--batch_size 128 \
--eval_batch_size 64 \
--lr 0.0008 \
--kl_start_step 2000 \
--kl_end_step 30000 \
--gradient_accumulation 1 \
--bbox_format ltwh \
--discrete_x_grid 128 \
--discrete_y_grid 128

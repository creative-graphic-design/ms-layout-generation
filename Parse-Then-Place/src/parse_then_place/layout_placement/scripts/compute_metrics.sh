#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_PATH="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_SRC_DIR="$(cd "${BASE_PATH}/../.." && pwd)"





data_dir=$1
result_write_to=$2
dataset_name=$3

if [[ ! -d ${result_write_to} ]]
then
    mkdir -p ${result_write_to}
fi

PYTHONPATH="${PROJECT_SRC_DIR}" python ${BASE_PATH}/eval.py \
--train_source_file train.json \
--test_ground_truth_file test.json \
--data_dir ${data_dir} \
--out_dir ${result_write_to} \
--dataset_name ${dataset_name}

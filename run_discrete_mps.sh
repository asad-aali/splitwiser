#!/bin/bash
function enable_mps_if_needed()
{
    echo "Enabling MPS"
    nvidia-smi -i 3 -c EXCLUSIVE_PROCESS
    nvidia-cuda-mps-control -d

    if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 1 ]]; then
        echo "Unable to enable MPS"
        exit 1
    fi
}

function disable_mps_if_needed()
{
    echo quit | nvidia-cuda-mps-control
    nvidia-smi -i 3 -c DEFAULT

    if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 0 ]]; then
        echo "Unable to disable MPS"
        exit 1
    fi
}

enable_mps_if_needed

model=opt-125m
case_id=0
CUDA_VISIBLE_DEVICES=0 python src/run_discrete_mp.py \
    --model $model \
    --case_id $case_id
    
disable_mps_if_needed
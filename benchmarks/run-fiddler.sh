#!/bin/bash

echo "Running Fiddler for OpenAI HumanEval benchmark..."
echo "NOTE: Make sure you have the required environment set up."
echo "    1. Please replace CONDSA_BASE_DIR with the path to your conda installation"
echo "    2. Please replace --target-model and --draft-model with the correct model path if you have a local copy."
echo "    3. The hyperparameters are set for developer's machine. You may need to adjust them according to your machine's configuration."

# Activate the conda environment
CONDA_BASE_DIR=~/miniconda3/
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
conda activate fiddler
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

modelType=$1
dataset=$2
if [ -z "$dataset" ]; then
    echo "Usage: $0 <modelType> <dataset>"
    exit 1
fi
if [ "$dataset" == "openai_humaneval" ]; then
    if [ "$modelType" == "8x7B" ]; then
        python benchmarks/baseline/fiddler.py --model mistralai/Mixtral-8x7B-v0.1 \
            --batch-size 139 --n-token 32 --dataset openai_humaneval
    elif [ "$modelType" == "8x22B" ]; then
        python benchmarks/baseline/fiddler.py --model mistralai/Mixtral-8x22B-v0.1 \
            --batch-size 64 --n-token 32 --dataset openai_humaneval
    else
            echo "Unsupported model type. Use '8x7B' or '8x22B'."
            exit 1
    fi
elif [ "$dataset" == "samsum" ]; then
    if [ "$modelType" == "8x7B" ]; then
        python benchmarks/baseline/fiddler.py --model mistralai/Mixtral-8x7B-v0.1 \
            --batch-size 81 --n-token 32 --dataset samsum
    elif [ "$modelType" == "8x22B" ]; then
        python benchmarks/baseline/fiddler.py --model mistralai/Mixtral-8x22B-v0.1 \
            --batch-size 40 --n-token 32 --dataset samsum
    else
        echo "Unsupported model type. Use '8x7B' or '8x22B'."
        exit 1
    fi
elif [ "$dataset" == "summeval" ]; then
    if [ "$modelType" == "8x7B" ]; then
        python benchmarks/baseline/fiddler.py --model mistralai/Mixtral-8x7B-v0.1 \
            --batch-size 61 --n-token 32 --dataset summeval
    elif [ "$modelType" == "8x22B" ]; then
        python benchmarks/baseline/fiddler.py --model mistralai/Mixtral-8x22B-v0.1 \
            --batch-size 32 --n-token 32 --dataset summeval
    else
        echo "Unsupported model type. Use '8x7B' or '8x22B'."
        exit 1
    fi
elif [ "$dataset" == "ceval_exam" ]; then
    if [ "$modelType" == "8x7B" ]; then
        python benchmarks/baseline/fiddler.py --model mistralai/Mixtral-8x7B-v0.1 \
            --batch-size 118 --n-token 32 --dataset ceval_exam
    elif [ "$modelType" == "8x22B" ]; then
        python benchmarks/baseline/fiddler.py --model mistralai/Mixtral-8x22B-v0.1 \
            --batch-size 64 --n-token 32 --dataset ceval_exam
    else
        echo "Unsupported model type. Use '8x7B' or '8x22B'."
        exit 1
    fi
else
    echo "Unsupported dataset. Use 'openai_humaneval', 'samsum', 'summeval' or 'ceval_exam'."
    exit 1
fi

#!/bin/bash

echo "Running SpecOffload for OpenAI HumanEval benchmark..."
echo "NOTE: Make sure you have the required environment set up."
echo "    1. Please replace CONDSA_BASE_DIR with the path to your conda installation"
echo "    2. Please replace --target-model and --draft-model with the correct model path if you have a local copy."
echo "    3. The hyperparameters are set for developer's machine. You may need to adjust them according to your machine's configuration."

# Activate the conda environment
CONDA_BASE_DIR=~/miniconda3/
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
conda activate specoffload

modelType=$1
dataset=$2
if [ -z "$dataset" ]; then
    echo "Usage: $0 <modelType> <dataset>"
    exit 1
fi
if [ "$dataset" == "openai_humaneval" ]; then
    if [ "$modelType" == "8x7B" ]; then
        python examples/run.py \
                --target-model mistralai/Mixtral-8x7B-v0.1 \
                --draft-model mistralai/Mistral-7B-Instruct-v0.2 \
                --dataset openai_humaneval --cuda 0 \
                --n-token 32 \
                --generation-batch-size 256 \
                --prefill-batch-size 80 \
                --assisant-batch-size 10 \
                --assisant-max-new-tokens 6
    elif [ "$modelType" == "8x22B" ]; then
        python examples/run.py \
                --target-model mistralai/Mixtral-8x22B-v0.1 \
                --draft-model mistralai/Mistral-7B-Instruct-v0.2 \
                --dataset openai_humaneval --cuda 0 \
                --n-token 32 \
                --generation-batch-size 128 \
                --prefill-batch-size 32 \
                --assisant-batch-size 6 \
                --assisant-max-new-tokens 4
    else
        echo "Unsupported model type. Use '8x7B' or '8x22B'."
        exit 1
    fi
elif [ "$dataset" == "samsum" ]; then
    if [ "$modelType" == "8x7B" ]; then
        python examples/run.py \
                --target-model mistralai/Mixtral-8x7B-v0.1 \
                --draft-model mistralai/Mistral-7B-Instruct-v0.2 \
                --dataset samsum --cuda 0 \
                --n-token 32 \
                --generation-batch-size 256 \
                --prefill-batch-size 96 \
                --assisant-batch-size 10 \
                --assisant-max-new-tokens 4
    elif [ "$modelType" == "8x22B" ]; then
        python examples/run.py \
                --target-model mistralai/Mixtral-8x22B-v0.1 \
                --draft-model mistralai/Mistral-7B-Instruct-v0.2 \
                --dataset samsum --cuda 0 \
                --n-token 32 \
                --generation-batch-size 64 \
                --prefill-batch-size 16 \
                --assisant-batch-size 8 \
                --assisant-max-new-tokens 6
    else
        echo "Unsupported model type. Use '8x7B' or '8x22B'."
        exit 1
    fi
elif [ "$dataset" == "summeval" ]; then
    if [ "$modelType" == "8x7B" ]; then
        python examples/run.py \
                --target-model mistralai/Mixtral-8x7B-v0.1 \
                --draft-model mistralai/Mistral-7B-Instruct-v0.2 \
                --dataset summeval --cuda 0 \
                --n-token 32 \
                --generation-batch-size 320 \
                --prefill-batch-size 80 \
                --assisant-batch-size 8 \
                --assisant-max-new-tokens 8
    elif [ "$modelType" == "8x22B" ]; then
        python examples/run.py \
                --target-model mistralai/Mixtral-8x22B-v0.1 \
                --draft-model mistralai/Mistral-7B-Instruct-v0.2 \
                --dataset summeval --cuda 0 \
                --n-token 32 \
                --generation-batch-size 64 \
                --prefill-batch-size 16 \
                --assisant-batch-size 8 \
                --assisant-max-new-tokens 8
    else
        echo "Unsupported model type. Use '8x7B' or '8x22B'."
        exit 1
    fi
elif [ "$dataset" == "ceval_exam" ]; then
    if [ "$modelType" == "8x7B" ]; then
        python examples/run.py \
                --target-model mistralai/Mixtral-8x7B-v0.1 \
                --draft-model mistralai/Mistral-7B-Instruct-v0.2 \
                --dataset ceval_exam --cuda 0 \
                --n-token 32 \
                --generation-batch-size 256 \
                --prefill-batch-size 80 \
                --assisant-batch-size 10 \
                --assisant-max-new-tokens 6
    elif [ "$modelType" == "8x22B" ]; then
        python examples/run.py \
                --target-model mistralai/Mixtral-8x22B-v0.1 \
                --draft-model mistralai/Mistral-7B-Instruct-v0.2 \
                --dataset ceval_exam --cuda 0 \
                --n-token 32 \
                --generation-batch-size 128 \
                --prefill-batch-size 32 \
                --assisant-batch-size 6 \
                --assisant-max-new-tokens 4
    else
        echo "Unsupported model type. Use '8x7B' or '8x22B'."
        exit 1
    fi
else
    echo "Unsupported dataset. Use 'openai_humaneval', 'samsum', 'summeval' or 'ceval_exam'."
    exit 1
fi

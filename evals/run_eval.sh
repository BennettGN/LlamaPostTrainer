#!/bin/bash

#This part is need for OSC users
export CC=gcc
export CXX=g++
export TRITON_CACHE_DIR=/fs/scratch/xxx/${USER}/triton_cache


export UV_CACHE_DIR=/fs/scratch/xxx/owos/.cache/uv  #control your uv caches

export OPENAI_API_KEY="sk-dummy-not-used"



# git clone https://github.com/owos/olmes.git
# cd olmes

# uv sync --group gpu 


# cd evals/olmes/oe_eval/dependencies/safety
# uv pip install -e safety-eval 
# uv pip install -r safety-eval/requirements.txt



dataset_name=(
    "gsm8k"
    "mbpp"
    "ifeval"
    "xstest"
    # Local-model safety evals (no OpenAI required):
    # "harmbench::wildguard_reasoning_answer"
    # "xstest::wildguard_reasoning_answer"
    # "toxigen::tiny"  

)
model_path=allenai/OLMo-2-0425-1B-SFT

for dataset in "${dataset_name[@]}"; do
    echo "Evaluating on ${dataset}..."

    uv run olmes \
        --model ${model_path} \
        --task ${dataset} \
        --output-dir $model_path-eval-${dataset} 
done
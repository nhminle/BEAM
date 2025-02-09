#!/bin/bash
#SBATCH -J fireworks_prefixprobe_llama405b
#SBATCH --partition=cpu
#SBATCH --time=0-18:00:00
#SBATCH -o %x_%j.out
#SBATCH --open-mode=truncate

# Define the arrays
# module load cuda/11.8    # Don't change
source .venv/bin/activate
# conda activate vllm-env
python3 dir_probe_eval.py
# or meta-llama/Meta-Llama-3.1-8B-Instruct || allenai/OLMo-7B-0724-Instruct-hf
#!/bin/bash
#SBATCH -J prefixprove
#SBATCH --partition=cpu
#SBATCH --time=0-18:00:00
#SBATCH -o %x_%j.out
#SBATCH --open-mode=truncate
#SBATCH --mail-type=ALL       # notify you about the start and end of the job
#SBATCH --mail-user=ekorukluoglu@umass.edu # Email to which notifications will be sent
 
# Define the arrays
#module load cuda/11.8    # Don't change
source .venv/bin/activate
# conda activate vllm-env
python3 Evaluation/prefix_probe/prefix_probe_eval.py
# or meta-llama/Meta-Llama-3.1-8B-Instruct || allenai/OLMo-7B-0724-Instruct-hf
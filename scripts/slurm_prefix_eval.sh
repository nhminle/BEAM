#!/bin/bash
#SBATCH -J prefixprove
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:2080ti:1
#SBATCH --time=0-18:00:00
#SBATCH -o %x_%j.out
#SBATCH --open-mode=truncate
#SBATCH --mail-type=ALL       # notify you about the start and end of the job
#SBATCH --mail-user=ekorukluoglu@umass.edu # Email to which notifications will be sent
#SBATCH --array=0-4

# Load any required modules here
module load cuda/12.1

# Activate the virtual environment
source .venv/bin/activate

# Define the models array
models=('gpt-4o-2024-11-20' 'Llama-3.1-70B-Instruct' 'Llama-3.1-70B-Instruct-quantized.w4a16' 'Llama-3.1-70B-Instruct-quantized.w8a16')

# Get the model corresponding to this SLURM array index
model=${models[$SLURM_ARRAY_TASK_ID]}

# Execute the Python evaluation script with the selected model as an argument
python3 Evaluation/prefix_probe/prefix_probe_eval.py --model "$model"

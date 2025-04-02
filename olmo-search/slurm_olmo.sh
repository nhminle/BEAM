#!/bin/bash
#SBATCH -J olmocheck-go
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=70  
#SBATCH --time=0-48:00:00
#SBATCH -o %x_%j.out
#SBATCH --open-mode=truncate
#SBATCH --mail-type=ALL       # notify you about the start and end of the job
#SBATCH --mail-user=ekorukluoglu@umass.edu # Email to which notifications will be sent
 
# Define the arrays
# module load cuda/11.8    # Don't change
source /home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/.venv/bin/activate
# conda activate vllm-env
python3 searcher.py
# go run searcher.go
# or meta-llama/Meta-Llama-3.1-8B-Instruct || allenai/OLMo-7B-0724-Instruct-hf
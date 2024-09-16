#!/bin/bash
#SBATCH --job-name=ner_proc_test    # Anything
#SBATCH --partition=gpu-preempt    # Same
#SBATCH --mem=50G              # Same
#SBATCH --time=24:00:00         # Anything
#SBATCH --gres=gpu:2080ti:1       # Number of GPUs (gpu:name:number of gpus)
#SBATCH --output=log.out       # Standard output and error log
#SBATCH --mail-type=ALL         # notify you about the start and end of the job
#SBATCH --mail-user= # Email to which notifications will be sent
 
# Define the arrays
module load cuda/11.8.0         # Don't change
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ner
python3 process_txt_other_lang.py        # Run the script
#!/bin/bash
#SBATCH --job-name=olmocheck_binary
#SBATCH --partition=cpu
#SBATCH --output=global_shard_%A_%a.out
#SBATCH --error=global_shard_%A_%a.err
#SBATCH --open-mode=truncate
#SBATCH --mail-type=ALL       # notify you about the start and end of the job
#SBATCH --mail-user=ekorukluoglu@umass.edu # Email to which notifications will be sent
#SBATCH --time=0-48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
##### SBATCH --array=1-10



# Each job in the array gets its index via SLURM_ARRAY_TASK_ID.
# Pass that index to the binary.
# ./searcher_bin -index=${SLURM_ARRAY_TASK_ID}
go run searcher_go_old.go
#!/bin/bash
#SBATCH --output=./slurm_logs/slurm_logs_%A_%a.out
#SBATCH --job-name=gs0_lwta_dropout
#SBATCH --cpus-per-task=14
#SBATCH --array=0-100
#               d-hh:mm:ss
#SBATCH --time=14-00:00:00
ENV=/home/carta/hpc2019_experiments/latest_image/config.env
CWD=/home/carta/cl_code/
CONTAINER=/home/carta/cl_code/container/antonio.cl_image
MAIN=/home/carta/cl_code/recurrent_continual_learning/speech/train_incremental_gsc.py
ch-run --set-env=$ENV --cd=$CWD $CONTAINER /opt/conda/bin/python $MAIN -- --id="$SLURM_ARRAY_TASK_ID"

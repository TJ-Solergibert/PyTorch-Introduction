#!/bin/bash

#SBATCH --job-name benchmark
#SBATCH --chdir /home/upc/upc580327/lab1
#SBATCH --output reports/exp7/R-%x.%j.out
#SBATCH --error reports/exp7/R-%x.%j.err
#SBATCH --nodes 1                   # number of Nodes
#SBATCH --ntasks-per-node 1         # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:1                # Number of GPUs
#SBATCH --cpus-per-task 20          # number of CPUs per task. In MN5 must be Number of GPUs * 20
#SBATCH --time 01:29:59             # maximum execution time (DD-HH:MM:SS). Mandatory field in MN5
#SBATCH --account bsc98
#SBATCH --qos acc_bsccs
#SBATCH --exclusive

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

######################
### Set environment ###
######################
module purge
module load singularity
######################
MODEL=vit
DS=/home/upc/upc580327/lab1/micro-224

EPOCHS=5
BS=128
NW=10
OPTIM=adamw
LOG_ITER=500
EPOCHS_EVAL_FREQ=5
# Faltaria compile & mixed precision
PYTHON_FILE=/home/upc/upc580327/lab1/train.py
PYTHON_ARGS="--model_name $MODEL \
    --dataset $DS \
    --num_epochs $EPOCHS \
    --batch_size $BS \
    --eval_batch_size $BS \
    --num_workers $NW \
    --optimizer $OPTIM \
    --iteration_logging $LOG_ITER \
    --epochs_eval $EPOCHS_EVAL_FREQ \
    --mixed_precision bf16 \
    --compile \
    "

export CMD="python3 $PYTHON_FILE $PYTHON_ARGS"

SINGULARITY_CONTAINER=/gpfs/apps/MN5/ACC/SINGULARITY/SRC/images/nvidiaPytorch24.07
SINGULARITY_ARGS=" \
    --bind /home/upc/upc580327/lab1 \
    --nv \
    $SINGULARITY_CONTAINER \
    "

SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    "

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bsc_singularity exec  $SINGULARITY_ARGS bash -c "$CMD"

echo "END TIME: $(date)"

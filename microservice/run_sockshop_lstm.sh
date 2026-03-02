#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=sockshop_lstm
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
set -euo pipefail

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
DATA=$SCRATCH/micro-service-trace-data/preprocessed

cd $PROJECT
mkdir -p logs

module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source $PROJECT/.venv/bin/activate

srun nvidia-smi

srun python microservice/train_sockshop.py \
    --data_path  "$DATA" \
    --log_folder logs/sockshop-lstm-1 \
    --model      lstm \
    --train_split train_id \
    --valid_split valid_id \
    --test_split  test_id \
    --ood_valid_splits "valid_ood_cpu,valid_ood_disk,valid_ood_mem" \
    --ood_test_splits  "test_ood_cpu,test_ood_disk,test_ood_mem" \
    --n_categories  6 \
    --n_hidden  256 \
    --n_layer   2 \
    --dim_sys   48 \
    --dim_proc  48 \
    --dim_entry 12 \
    --dim_ret   12 \
    --dim_pid   12 \
    --dim_tid   12 \
    --dim_time  12 \
    --dim_order 12 \
    --dim_f_mean 0 \
    --n_update  1000000 \
    --eval      1000 \
    --lr        0.001 \
    --ls        0.1 \
    --batch     32 \
    --clip      10 \
    --dropout   0.01 \
    --gpu       "0" \
    --amp \
    --reduce_lr_patience      5 \
    --early_stopping_patience 20 \
    --seed      1 \
    --train_event_model \
    --train_latency_model \
    --analysis

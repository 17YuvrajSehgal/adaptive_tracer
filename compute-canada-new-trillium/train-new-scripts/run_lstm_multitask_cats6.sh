#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=lstm_multitask_cats6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Only these 2 lines change across all 12 jobs ────────────────────────────
CATS=6        # folder suffix: preprocessed_cats6
N_CAT=6       # must match --n_categories used during preprocessing
# ────────────────────────────────────────────────────────────────────────────

module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
DATA=$SCRATCH/micro-service-trace-data/preprocessed_cats${CATS}
LOG_DIR=$PROJECT/logs/lstm_multitask_cats${CATS}_${SLURM_JOB_ID}

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline
export WANDB_DIR="$LOG_DIR"
export WANDB_CACHE_DIR="$SCRATCH/wandb_cache"
export TRITON_CACHE_DIR="$SCRATCH/.triton_cache"
export TORCH_HOME="$SCRATCH/.torch"
export HF_HOME="$SCRATCH/.hf_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME"

echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Mode         : LSTM  |  Event-only  |  cats=${CATS}"
echo "Data dir     : $DATA"
echo "Log dir      : $LOG_DIR"
echo "============================================================"

source "$PROJECT/.venv/bin/activate"

srun nvidia-smi

srun python -u microservice/train_sockshop.py \
    --preprocessed_dir  "$DATA" \
    --model             lstm \
    --n_categories      $N_CAT \
    --max_seq_len       512 \
    --n_hidden          1024 \
    --n_layer           6 \
    --dropout           0.1 \
    --dim_sys           48 \
    --dim_entry         12 \
    --dim_ret           12 \
    --dim_proc          48 \
    --dim_pid           12 \
    --dim_tid           12 \
    --dim_order         12 \
    --dim_time          12 \
    --train_event_model \
    --train_latency_model \
    --ood_score         combined \
    --batch             512 \
    --accum_steps       4 \
    --n_epochs          20 \
    --lr                3e-4 \
    --warmup_steps      500 \
    --clip              1.0 \
    --num_workers       4 \
    --label_smoothing   0.0 \
    --amp \
    --eval_every        200 \
    --save_every        5000 \
    --lat_score_weight  0.3 \
    --wandb_project     sockshop_lmat \
    --wandb_run_name    "lstm_multitask_cats${CATS}_${SLURM_JOB_ID}" \
    --log_dir           "$LOG_DIR" \
    --gpu               0

EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE"
echo "OOD results: $LOG_DIR/ood_results.json"
wandb sync "$LOG_DIR/wandb/latest-run" 2>/dev/null || true
exit $EXIT_CODE

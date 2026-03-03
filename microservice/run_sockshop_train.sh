#!/bin/bash
# =============================================================================
# run_sockshop_train.sh
# Submit LMAT training on a single H100 GPU (Narval / CC cluster)
# =============================================================================
#SBATCH --job-name=sockshop_train
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --partition=compute_full_node
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load StdEnv/2023 python/3.11.5

PROJECT=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer
source $PROJECT/.venv/bin/activate

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PREPROCESSED=$SCRATCH/micro-service-trace-data/preprocessed
LOG_DIR=$SCRATCH/adaptive_tracer/logs/sockshop_${SLURM_JOB_ID}
WANDB_DIR=$LOG_DIR

mkdir -p "$LOG_DIR"

# ── WandB (offline mode is safe on cluster — sync later with `wandb sync`) ──
export WANDB_MODE=offline
export WANDB_DIR="$LOG_DIR"

echo "============================================================"
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURMD_NODENAME"
echo "GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Preprocessed: $PREPROCESSED"
echo "Log dir     : $LOG_DIR"
echo "============================================================"

# ── Training ─────────────────────────────────────────────────────────────────
cd $PROJECT

python -u microservice/train_sockshop.py \
    --preprocessed_dir "$PREPROCESSED" \
    \
    --model transformer \
    --n_head    8 \
    --n_hidden  1024 \
    --n_layer   6 \
    --dropout   0.1 \
    --activation gelu \
    --dim_sys   64 \
    --dim_entry  8 \
    --dim_ret    8 \
    --dim_proc   8 \
    --dim_pid   16 \
    --dim_tid   16 \
    --dim_order 16 \
    --dim_time  16 \
    \
    --train_event_model \
    --train_latency_model \
    --n_categories 6 \
    \
    --batch        512 \
    --accum_steps    4 \
    --n_epochs      20 \
    --lr          3e-4 \
    --warmup_steps 2000 \
    --clip          1.0 \
    --num_workers     4 \
    --label_smoothing 0.1 \
    \
    --amp \
    --compile \
    \
    --eval_every  2000 \
    --save_every  5000 \
    \
    --wandb_project sockshop_lmat \
    --wandb_run_name "transformer_h100_${SLURM_JOB_ID}" \
    --log_dir "$LOG_DIR" \
    --gpu 0

EXIT_CODE=$?
echo "============================================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "Log dir: $LOG_DIR"
echo "OOD results: $LOG_DIR/ood_results.json"

# Sync WandB offline run once training is done
wandb sync "$LOG_DIR/wandb/latest-run" 2>/dev/null || true

echo "============================================================"
exit $EXIT_CODE

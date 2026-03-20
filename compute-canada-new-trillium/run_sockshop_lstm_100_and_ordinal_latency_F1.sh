#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=sockshop_lstm_f1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ---- Modules (per Trillium docs) ----
module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

# ---- Paths ----
SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
DATA=$SCRATCH/micro-service-trace-data/preprocessed
LOG_DIR=$PROJECT/logs/sockshop_lstm_100ep_ordinal_F1_${SLURM_JOB_ID}

mkdir -p "$LOG_DIR"

# ---- Optional: avoid CPU thread blowups ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- WandB offline (safe on cluster) ----
export WANDB_MODE=offline
export WANDB_DIR="$LOG_DIR"
export WANDB_CACHE_DIR="$SCRATCH/wandb_cache"

# ---- Redirect caches that would otherwise write to read-only /home ----
export TRITON_CACHE_DIR="$SCRATCH/.triton_cache"
export TORCH_HOME="$SCRATCH/.torch"
export HF_HOME="$SCRATCH/.hf_cache"

mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME"

echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "Node         : $SLURMD_NODENAME"
echo "Submit dir   : $SLURM_SUBMIT_DIR"
echo "Project dir  : $PROJECT"
echo "Scratch dir  : $SCRATCH"
echo "Preprocessed : $DATA"
echo "Log dir      : $LOG_DIR"
echo "============================================================"

echo "[1/6] Module list"
module list

echo "[2/6] GPU visible to job"
srun nvidia-smi

echo "[3/6] Python + CUDA sanity (PyTorch)"
source "$PROJECT/.venv/bin/activate"

srun python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("matmul ok, y mean:", y.mean().item())
PY

echo "[4/6] Check that preprocessed directory exists"
srun bash -lc "ls -lah '$DATA' | head -30"

cd "$PROJECT"

python -u microservice/train_sockshop.py \
    --preprocessed_dir "$DATA" \
    --model     lstm \
    --max_seq_len 4096 \
    --n_hidden  1024 \
    --n_layer   6 \
    --dropout   0.01 \
    --dim_sys   48 \
    --dim_entry 12 \
    --dim_ret   12 \
    --dim_proc  48 \
    --dim_pid   12 \
    --dim_tid   12 \
    --dim_order 12 \
    --dim_time  12 \
    --train_event_model \
    --train_latency_model \
    --ood_score combined \
    --ood_threshold_grid 100 \
    --n_categories 6 \
    --batch        512 \
    --accum_steps    4 \
    --n_epochs     100 \
    --lr          1e-3 \
    --warmup_steps 2000 \
    --clip          10.0 \
    --num_workers     4 \
    --label_smoothing 0.1 \
    --amp \
    --eval_every  100 \
    --save_every  5000 \
    --ordinal_latency \
    --lat_score_weight 0.3 \
    --wandb_project sockshop_lmat \
    --wandb_run_name "lstm_h100_100ep_ordinal_F1_${SLURM_JOB_ID}" \
    --log_dir "$LOG_DIR" \
    --gpu 0

EXIT_CODE=$?
echo "============================================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "Log dir: $LOG_DIR"
echo "OOD results: $LOG_DIR/ood_results.json"

# Sync WandB offline run once training is done (optional)
wandb sync "$LOG_DIR/wandb/latest-run" 2>/dev/null || true

echo "============================================================"
exit $EXIT_CODE


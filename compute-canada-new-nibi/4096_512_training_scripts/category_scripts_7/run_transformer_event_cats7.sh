#!/bin/bash
#SBATCH --job-name=trf_event_cats7_4096_512
#SBATCH --account=def-naser2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

BASE_SCRATCH=${SCRATCH:-$HOME/scratch}
WORKROOT=$BASE_SCRATCH/adaptive_tracing_scratch
PROJECT=$WORKROOT/adaptive_tracer
DATA=$WORKROOT/micro-service-trace-data/preprocessed_lmat_kernel_cats7_seq4096_512
LOG_DIR=$PROJECT/logs/transformer_event_cats7_seq4096_512_${SLURM_JOB_ID}

mkdir -p "$LOG_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WANDB_MODE=offline
export WANDB_DIR="$LOG_DIR"
export WANDB_CACHE_DIR="$WORKROOT/wandb_cache"
export TRITON_CACHE_DIR="$WORKROOT/.triton_cache"
export TORCH_HOME="$WORKROOT/.torch"
export HF_HOME="$WORKROOT/.hf_cache"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_HOME" "$WANDB_CACHE_DIR"

echo "============================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : ${SLURMD_NODENAME:-unknown}"
echo "Mode         : Transformer | Event-only | paper bins=7 | seq=4096_512"
echo "Project      : $PROJECT"
echo "Data dir     : $DATA"
echo "Log dir      : $LOG_DIR"
echo "============================================================"

source "$PROJECT/.venv/bin/activate"
cd "$PROJECT"

echo "[1/4] Module list"
module list

echo "[2/4] GPU sanity"
srun nvidia-smi

echo "[3/4] Python + torch sanity"
srun python - <<'PY'
import sys
import torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "[4/4] One-batch Transformer smoke test"
DATA_DIR="$DATA" srun python - <<'PY'
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from microservice.NpzDataset import SockshopNpzDataset, sockshop_collate_fn
from microservice.train_sockshop import build_model, forward_batch, compute_loss

class Args:
    preprocessed_dir = os.environ["DATA_DIR"]
    n_categories = 8
    max_seq_len = 4096
    model = "transformer"
    n_head = 8
    n_hidden = 1024
    n_layer = 6
    dropout = 0.1
    activation = "gelu"
    tfixup = False
    dim_sys = 48
    dim_entry = 12
    dim_ret = 12
    dim_proc = 48
    dim_pid = 12
    dim_tid = 12
    dim_order = 12
    dim_time = 12
    dim_f_mean = 0
    train_event_model = True
    train_latency_model = False
    ordinal_latency = False
    multitask_lambda = 0.5
    chk = False
    amp = False
    label_smoothing = 0.0

with open(os.path.join(Args.preprocessed_dir, "vocab.pkl"), "rb") as f:
    dict_sys, dict_proc = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ds = SockshopNpzDataset(
    os.path.join(Args.preprocessed_dir, "train_id"),
    batch_size=2,
    max_seq_len=Args.max_seq_len,
    max_samples=2,
    shuffle_shards=False,
)
loader = DataLoader(ds, batch_size=None, collate_fn=sockshop_collate_fn, num_workers=0)
batch = next(iter(loader))
model = build_model(Args, len(dict_sys), len(dict_proc), device)
logits_e, logits_l = forward_batch(model, batch, device, Args)
crit_e = nn.CrossEntropyLoss(ignore_index=0)
loss, loss_e, loss_l = compute_loss(logits_e, logits_l, batch, device, Args, crit_e, None)
print("event logits:", tuple(logits_e.shape))
print("latency logits:", tuple(logits_l.shape))
print("loss:", float(loss.item()), float(loss_e.item()), float(loss_l.item()))
PY

python -u microservice/train_sockshop.py \
  --preprocessed_dir "$DATA" \
  --model transformer \
  --n_categories 8 \
  --max_seq_len 4096 \
  --n_head 8 \
  --n_hidden 1024 \
  --n_layer 6 \
  --dropout 0.1 \
  --activation gelu \
  --dim_sys 48 \
  --dim_entry 12 \
  --dim_ret 12 \
  --dim_proc 48 \
  --dim_pid 12 \
  --dim_tid 12 \
  --dim_order 12 \
  --dim_time 12 \
  --train_event_model \
  --ood_score event \
  --batch 4 \
  --accum_steps 16 \
  --n_epochs 100 \
  --early_stopping_patience 20 \
  --lr 3e-4 \
  --warmup_steps 500 \
  --clip 1.0 \
  --num_workers 8 \
  --label_smoothing 0.1 \
  --amp \
  --eval_every 200 \
  --save_every 5000 \
  --lat_score_weight 0.0 \
  --wandb_project sockshop_lmat \
  --wandb_run_name "transformer_event_cats7_seq4096_512_${SLURM_JOB_ID}" \
  --log_dir "$LOG_DIR" \
  --gpu 0

EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE"
echo "OOD results: $LOG_DIR/ood_results.json"
wandb sync "$LOG_DIR/wandb/latest-run" 2>/dev/null || true
exit $EXIT_CODE

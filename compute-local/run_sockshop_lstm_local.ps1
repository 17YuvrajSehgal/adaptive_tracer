# ============================================================
# run_sockshop_lstm_local.ps1
# Local GPU training — LSTM, ordinal latency (Apache-style OOD metrics in train_sockshop)
# Mirrors: compute-canada-new-trillium/run_sockshop_lstm_100_and_ordinal_latency.sh
#
# Default: 1 epoch smoke test. For full training, set:
#   $N_EPOCHS = 100
#   $EVAL_EVERY = 100
#   $SAVE_EVERY = 5000
#
# Usage (from repo root or any directory):
#   .\compute-local\run_sockshop_lstm_local.ps1
#
# Prerequisites:
#   - .venv activated (or run: .\.venv\Scripts\Activate.ps1 first)
#   - Preprocessed SockShop data in $DATA below
# ============================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---- Run profile (smoke vs full) ----
$N_EPOCHS   = 1      # 100 for production run
$EVAL_EVERY = 50     # validation frequency in *optimizer steps*; lower helps 1-epoch smokes
$SAVE_EVERY = 500    # checkpoint interval; use 5000 for long runs

# ---- Paths ----
$PROJECT  = "c:\workplace\adaptive_tracer"
$DATA     = "c:\workplace\adaptive_tracer\micro-service-trace-data\preprocessed"
$RUN_TS   = (Get-Date -Format "yyyyMMdd_HHmmss")
$LOG_DIR  = "c:\workplace\adaptive_tracer\logs\sockshop_lstm_${N_EPOCHS}ep_ordinal_$RUN_TS"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# ---- Environment ----
$env:OMP_NUM_THREADS       = "1"
$env:MKL_NUM_THREADS       = "1"
$env:OPENBLAS_NUM_THREADS  = "1"
$env:NUMEXPR_NUM_THREADS   = "1"
$env:WANDB_MODE            = "offline"
$env:WANDB_DIR             = $LOG_DIR

# ---- Info ----
Write-Host "============================================================"
Write-Host "Script       : run_sockshop_lstm_local.ps1"
Write-Host "Project dir  : $PROJECT"
Write-Host "Preprocessed : $DATA"
Write-Host "Log dir      : $LOG_DIR"
Write-Host "Run timestamp: $RUN_TS"
Write-Host "Epochs       : $N_EPOCHS"
Write-Host "============================================================"

# ---- Sanity checks ----
Write-Host "[1/3] Python + CUDA sanity"
@'
import sys, torch
print("Python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu name:", torch.cuda.get_device_name(0))
    y = torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")
    print("matmul ok, y mean:", round(y.mean().item(), 4))
'@ | python

Write-Host "[2/3] Check that preprocessed directory exists"
if (-not (Test-Path $DATA)) {
    Write-Error "Preprocessed data not found at: $DATA"
    exit 1
}
Get-ChildItem $DATA | Format-Table Name, LastWriteTime

Write-Host "[3/3] Starting training"
Set-Location $PROJECT

python -u microservice/train_sockshop.py `
    --preprocessed_dir $DATA `
    --model     lstm `
    --max_seq_len 4096 `
    --n_hidden  1024 `
    --n_layer   6 `
    --dropout   0.01 `
    --dim_sys   48 `
    --dim_entry 12 `
    --dim_ret   12 `
    --dim_proc  48 `
    --dim_pid   12 `
    --dim_tid   12 `
    --dim_order 12 `
    --dim_time  12 `
    --train_event_model `
    --train_latency_model `
    --ordinal_latency `
    --ood_score combined `
    --n_categories 6 `
    --batch        64 `
    --accum_steps  32 `
    --n_epochs    $N_EPOCHS `
    --lr          1e-3 `
    --warmup_steps 2000 `
    --clip          10.0 `
    --num_workers     2 `
    --label_smoothing 0.1 `
    --amp `
    --eval_every  $EVAL_EVERY `
    --save_every  $SAVE_EVERY `
    --lat_score_weight 0.3 `
    --wandb_project sockshop_lmat `
    --wandb_run_name "lstm_local_${N_EPOCHS}ep_ordinal_$RUN_TS" `
    --log_dir $LOG_DIR `
    --gpu 0

$EXIT_CODE = $LASTEXITCODE
Write-Host "============================================================"
Write-Host "Training finished with exit code: $EXIT_CODE"
Write-Host "Log dir      : $LOG_DIR"
Write-Host "OOD results  : $LOG_DIR\ood_results.json"
Write-Host "============================================================"
exit $EXIT_CODE

# ============================================================
# run_lstm_event_cats5_local.ps1
# Local GPU training - LSTM, event, paper bins=5 (effective n_categories=6)
# Mirrors:
#   compute-canada-new-trillium/final-preprocessing/step-3-train-512-5000-models/
#   category_scripts_5/run_lstm_event_cats5.sh
#
# Only the settings most likely to exhaust an 8 GB local GPU are reduced:
#   - batch size
#   - num_workers
#
# Effective training/data/model settings otherwise stay aligned with the cluster script.
# ============================================================

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---- Paths ----
$PROJECT = "C:\workplace\adaptive_tracer"
$DATA = "C:\workplace\adaptive_tracer\micro-service-trace-data\preprocessed_seq512\preprocessed_lmat_kernel_cats5_seq512"
$RUN_TS = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = "C:\workplace\adaptive_tracer\logs\lstm_event_cats5_seq512_local_$RUN_TS"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# ---- Environment ----
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:WANDB_MODE = "offline"
$env:WANDB_DIR = $LOG_DIR

# ---- Info ----
Write-Host "============================================================"
Write-Host "Script       : run_lstm_event_cats5_local.ps1"
Write-Host "Mode         : LSTM | event | paper bins=5 | seq=512"
Write-Host "Project dir  : $PROJECT"
Write-Host "Preprocessed : $DATA"
Write-Host "Log dir      : $LOG_DIR"
Write-Host "Run timestamp: $RUN_TS"
Write-Host "GPU target   : Local RTX 5060 8 GB"
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
'@ | python

Write-Host "[2/3] Check that preprocessed directory exists"
if (-not (Test-Path $DATA)) {
    Write-Error "Preprocessed data not found at: $DATA"
    exit 1
}

Write-Host "[3/3] Starting training"
Set-Location $PROJECT

python -u microservice/train_sockshop.py `
    --preprocessed_dir $DATA `
    --model lstm `
    --n_categories 6 `
    --max_seq_len 512 `
    --n_hidden 1024 `
    --n_layer 6 `
    --dropout 0.1 `
    --dim_sys 48 `
    --dim_entry 12 `
    --dim_ret 12 `
    --dim_proc 48 `
    --dim_pid 12 `
    --dim_tid 12 `
    --dim_order 12 `
    --dim_time 12 `
    --train_event_model `
    --batch 16 `
    --accum_steps 4 `
    --n_epochs 1 `
    --early_stopping_patience 5 `
    --lr 3e-4 `
    --warmup_steps 500 `
    --clip 1.0 `
    --num_workers 2 `
    --label_smoothing 0.0 `
    --amp `
    --eval_every 1000 `
    --save_every 10000 `
    --wandb_project sockshop_lmat `
    --wandb_run_name "lstm_event_cats5_seq512_local_$RUN_TS" `
    --log_dir $LOG_DIR `
    --gpu 0

$EXIT_CODE = $LASTEXITCODE
Write-Host "============================================================"
Write-Host "Training finished with exit code: $EXIT_CODE"
Write-Host "Log dir      : $LOG_DIR"
Write-Host "OOD results  : $LOG_DIR\ood_results.json"
Write-Host "============================================================"
exit $EXIT_CODE

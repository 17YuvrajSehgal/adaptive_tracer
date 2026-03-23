#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=preprocess_train_cats4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TRACE_ROOT=$SCRATCH/micro-service-trace-data/traces
OUTPUT_DIR=$SCRATCH/micro-service-trace-data/preprocessed_cats4
TXT_DUMP=$SCRATCH/micro-service-trace-data/micro-service-trace-data-txt-dump

cd $PROJECT; mkdir -p logs
module purge; module load StdEnv/2023; module load python/3.11.5
module load babeltrace2 2>/dev/null || true
source $PROJECT/.venv/bin/activate

echo "[$(date +%T)] Starting preprocessing"

srun python -u microservice/preprocess_sockshop.py \
    --trace_root "$TRACE_ROOT" --output_dir "$OUTPUT_DIR" \
    --txt_dump_dir "$TXT_DUMP" --load_vocab "$OUTPUT_DIR" \
    --splits \
    "train_id:normal/run01:0" \
    "valid_id:normal/run01:0" \
    "test_id:normal/run01:0" \
    --split_ratios 0.70:0.15:0.15 \
    --seg_mode time --window_ms 100 --warmup_s 5 --min_events 8 \
    --max_seq_len 512 --n_categories 4 --shard_size 5000

echo "[$(date +%T)] done → $OUTPUT_DIR"
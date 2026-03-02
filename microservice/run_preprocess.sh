#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=sockshop_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TRACE_ROOT=$SCRATCH/micro-service-trace-data/traces
OUTPUT_DIR=$SCRATCH/micro-service-trace-data/preprocessed

# ── Environment ───────────────────────────────────────────────────────────────
cd $PROJECT
mkdir -p logs

module purge
module load StdEnv/2023
module load python/3.11.5

# babeltrace2 must be available for trace parsing
module load babeltrace2 2>/dev/null || true

source $PROJECT/.venv/bin/activate

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Adjust the split definitions below to match your actual run directories.
# Format: "split_name:run_dir1,run_dir2,...:label"
#   label = 0  → normal (in-distribution)
#   label = 1  → anomaly (out-of-distribution)
#
# The FIRST split is always treated as the training split (builds vocab + latency
# boundaries).  All subsequent splits use the frozen vocab.

srun python -u microservice/preprocess_sockshop.py \
    --trace_root  "$TRACE_ROOT" \
    --output_dir  "$OUTPUT_DIR" \
    --splits \
        "train_id:normal/run01,normal/run02,normal/run03:0" \
        "valid_id:normal/run04:0" \
        "test_id:normal/run05:0" \
        "valid_ood_cpu:cpu_stress/run01:1" \
        "test_ood_cpu:cpu_stress/run02:1" \
        "valid_ood_disk:disk_stress/run01:1" \
        "test_ood_disk:disk_stress/run02:1" \
        "valid_ood_mem:mem_stress/run01:1" \
        "test_ood_mem:mem_stress/run02:1" \
    --seg_mode  ust \
    --window_ms 100 \
    --warmup_s  5 \
    --min_events 8 \
    --max_seq_len 512 \
    --n_categories 6 \
    --shard_size 5000

echo "Preprocessing complete → $OUTPUT_DIR"

#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=sockshop_preprocess
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TRACE_ROOT=$SCRATCH/micro-service-trace-data/traces
OUTPUT_DIR=$SCRATCH/micro-service-trace-data/preprocessed

# Directory where babeltrace2 text dumps live (tree-structured):
#   $TXT_DUMP/normal/run01/kernel.txt
#   $TXT_DUMP/anomaly_cpu/ultra_01/kernel.txt   ... etc.
# Produce these BEFORE submitting this job by running on a login node:
#
#   for TYPE in anomaly_cpu anomaly_disk anomaly_mem anomaly_net normal; do
#     for RUN_DIR in "$TRACE_ROOT/$TYPE"/*/; do
#       RUN=$(basename "$RUN_DIR")
#       mkdir -p "$TXT_DUMP/$TYPE/$RUN"
#       babeltrace2 "$RUN_DIR/kernel" > "$TXT_DUMP/$TYPE/$RUN/kernel.txt"
#       babeltrace2 "$RUN_DIR/ust"    > "$TXT_DUMP/$TYPE/$RUN/ust.txt"
#     done
#   done
TXT_DUMP=$SCRATCH/micro-service-trace-data/micro-service-trace-data-txt-dump

# ── Environment ───────────────────────────────────────────────────────────────
cd $PROJECT
mkdir -p logs

module purge
module load StdEnv/2023
module load python/3.11.5
module load babeltrace2 2>/dev/null || true

source $PROJECT/.venv/bin/activate

# ── Check that text dumps exist ───────────────────────────────────────────────
echo "[$(date +%T)] Checking text dump directory: $TXT_DUMP"
if [ ! -d "$TXT_DUMP" ]; then
    echo "[ERROR] txt_dump_dir not found: $TXT_DUMP"
    echo "        Run the babeltrace2 dump loop on the login node first."
    exit 1
fi

# ── Preprocessing (reads text dumps — fast path, ~5-10x faster than bt2 API) ─
# One run per scenario; normal/run01 is reused for train/valid/test.
# anomaly_net is a 4th anomaly type.

srun python -u microservice/preprocess_sockshop.py \
    --trace_root    "$TRACE_ROOT" \
    --output_dir    "$OUTPUT_DIR" \
    --txt_dump_dir  "$TXT_DUMP" \
    --splits \
        "train_id:normal/run01:0" \
        "valid_id:normal/run01:0" \
        "test_id:normal/run01:0" \
        "valid_ood_cpu:anomaly_cpu/ultra_01:1" \
        "test_ood_cpu:anomaly_cpu/ultra_01:1" \
        "valid_ood_disk:anomaly_disk/ultra_01:1" \
        "test_ood_disk:anomaly_disk/ultra_01:1" \
        "valid_ood_mem:anomaly_mem/ultra_01:1" \
        "test_ood_mem:anomaly_mem/ultra_01:1" \
        "valid_ood_net:anomaly_net/ultra_01:1" \
        "test_ood_net:anomaly_net/ultra_01:1" \
    --seg_mode   time \
    --window_ms  100 \
    --warmup_s   5 \
    --min_events 8 \
    --max_seq_len 512 \
    --n_categories 6 \
    --shard_size 5000

echo "Preprocessing complete → $OUTPUT_DIR"

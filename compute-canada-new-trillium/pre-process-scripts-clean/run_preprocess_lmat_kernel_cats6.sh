#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=prep_lmat_kernel_c6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TRACE_ROOT=$SCRATCH/micro-service-trace-data
OUTPUT_DIR=$SCRATCH/micro-service-trace-data/preprocessed_lmat_kernel_cats6
TXT_DUMP=$SCRATCH/micro-service-trace-data/micro-service-trace-data-txt-dump

cd "$PROJECT"
mkdir -p "$OUTPUT_DIR" logs

module purge
module load StdEnv/2023
module load python/3.11.5
module load babeltrace2 2>/dev/null || true

source "$PROJECT/.venv/bin/activate"

echo "[$(date +%T)] Starting LMAT kernel preprocessing"
echo "TRACE_ROOT=$TRACE_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"

srun python -u microservice/preprocess_lmat_kernel.py \
  --trace_root "$TRACE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --txt_dump_dir "$TXT_DUMP" \
  --window_ms 100 \
  --warmup_s 5 \
  --min_events 8 \
  --max_seq_len 512 \
  --n_categories 6 \
  --shard_size 5000 \
  --normal_dir normal \
  --cpu_dir anomaly_cpu \
  --disk_dir anomaly_disk \
  --mem_dir anomaly_mem \
  --net_dir anomaly_net \
  --normal_train_runs run01,run02,run03 \
  --normal_valid_runs run04 \
  --normal_test_runs run05 \
  --ood_valid_runs run04 \
  --ood_test_runs run05

echo "[$(date +%T)] Done -> $OUTPUT_DIR"

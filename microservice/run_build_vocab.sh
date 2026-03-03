#!/bin/bash
#SBATCH --account=def-naser2
#SBATCH --partition=compute
#SBATCH --job-name=build_vocab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Vocab-Build Job ────────────────────────────────────────────────────────────
# Reads normal/run01, builds vocab.pkl + delay_spans.pkl, then exits.
# No NPZ shards are written (--vocab_only).
#
# Once this job completes, launch all 5 split jobs in parallel:
#
#   JOB1=$(sbatch --parsable microservice/run_build_vocab.sh)
#   sbatch --dependency=afterok:$JOB1 microservice/run_preprocess_train.sh
#   sbatch --dependency=afterok:$JOB1 microservice/run_preprocess_anomaly_cpu.sh
#   sbatch --dependency=afterok:$JOB1 microservice/run_preprocess_anomaly_disk.sh
#   sbatch --dependency=afterok:$JOB1 microservice/run_preprocess_anomaly_mem.sh
#   sbatch --dependency=afterok:$JOB1 microservice/run_preprocess_anomaly_net.sh

set -euo pipefail

SCRATCH=/scratch/yuvraj17/adaptive_tracing_scratch
PROJECT=$SCRATCH/adaptive_tracer
TRACE_ROOT=$SCRATCH/micro-service-trace-data/traces
OUTPUT_DIR=$SCRATCH/micro-service-trace-data/preprocessed
TXT_DUMP=$SCRATCH/micro-service-trace-data/micro-service-trace-data-txt-dump

cd $PROJECT
mkdir -p logs "$OUTPUT_DIR"

module purge
module load StdEnv/2023
module load python/3.11.5
module load babeltrace2 2>/dev/null || true
source $PROJECT/.venv/bin/activate

echo "[$(date +%T)] Building vocabulary from normal/run01 ..."

srun python -u microservice/preprocess_sockshop.py \
    --trace_root   "$TRACE_ROOT" \
    --output_dir   "$OUTPUT_DIR" \
    --txt_dump_dir "$TXT_DUMP" \
    --splits       "train_id:normal/run01:0" \
    --seg_mode     time \
    --window_ms    100 \
    --warmup_s     5 \
    --min_events   8 \
    --max_seq_len  512 \
    --n_categories 6 \
    --shard_size   5000 \
    --vocab_only

echo "[$(date +%T)] Vocab build complete."
echo "  vocab.pkl      → $OUTPUT_DIR/vocab.pkl"
echo "  delay_spans.pkl → $OUTPUT_DIR/delay_spans.pkl"
echo "All 5 split jobs can now start in parallel."

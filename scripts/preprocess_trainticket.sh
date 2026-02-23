#!/bin/bash
# =============================================================================
# LMAT — Train-Ticket kernel trace preprocessing (CPU-only job)
#
# Reads LTTng kernel traces from the Train-Ticket microservice benchmark,
# segments them using per-PID 500 ms time windows, and writes NPZ shards
# ready for LMAT model training.
#
# Run this ONCE before any training job.  No GPU required.
#
# Estimated time:
#   Normal trace scan  (pass 1):  ~10-20 min  (timestamp-only scan)
#   Normal trace encode (pass 2): ~30-60 min  (full event decode + sharding)
#   7 anomaly traces (parallel):  ~30-60 min  (depends on n_workers)
#   Total:                        ~1-2 h
# =============================================================================

#SBATCH --account=def-naser2
#SBATCH --job-name=lmat_preprocess_tt

# --- Resources (no GPU) ------------------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # 1 for normal trace + up to 7 OOD workers + OS headroom
#SBATCH --mem=64G                  # babeltrace2 buffers several GB; scan + encode overlap
#SBATCH --time=03:00:00

# --- Output ------------------------------------------------------------------
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
set -euo pipefail

PROJECT=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer
TRACES=/scratch/yuvraj17/adaptive_tracing_scratch/lttng-traces-train-ticket
OUTPUT=/scratch/yuvraj17/adaptive_tracing_scratch/processed-train-ticket

cd "$PROJECT"
mkdir -p logs "$OUTPUT"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load StdEnv/2023
module load python/3.11.5
module load babeltrace2 2>/dev/null || true   # silently skip if not a module

source "$PROJECT/.venv/bin/activate"

# Confirm babeltrace2 is importable
python -c "import bt2; print('bt2 OK — version:', bt2.__version__)"

# ---------------------------------------------------------------------------
# OOD workers: use available CPUs minus 4 (OS + normal-trace headroom),
# capped at the number of anomaly traces (7).
# ---------------------------------------------------------------------------
N_ANOMALY=7
N_WORKERS=$(( SLURM_CPUS_PER_TASK - 4 ))
N_WORKERS=$(( N_WORKERS > N_ANOMALY ? N_ANOMALY : N_WORKERS ))
N_WORKERS=$(( N_WORKERS < 1 ? 1 : N_WORKERS ))
echo "Using ${N_WORKERS} parallel OOD workers"

# ---------------------------------------------------------------------------
# Run preprocessing
# ---------------------------------------------------------------------------
srun python -u data-preprocessing/preprocess_trainticket.py \
    \
    --normal_trace "$TRACES/train-ticket-normal-full/kernel" \
    \
    --anomaly_traces \
        "bandwidth:$TRACES/train-ticket-bandwidth/kernel,\
cpu_stress:$TRACES/train-ticket-cpu-stress/kernel,\
db_load:$TRACES/train-ticket-db-load/kernel,\
io_stress:$TRACES/train-ticket-io-stress/kernel,\
memory:$TRACES/train-ticket-memory/kernel,\
pod_restart:$TRACES/train-ticket-pod-restart/kernel,\
verbose_log:$TRACES/train-ticket-verbose-logging/kernel" \
    \
    --output_dir "$OUTPUT" \
    \
    --window_ms  500 \
    --min_events 10  \
    --train_frac 0.70 \
    --valid_frac 0.15 \
    --shard_size 5000 \
    --num_proc   "${N_WORKERS}"

# ---------------------------------------------------------------------------
# Quick sanity check — print output layout
# ---------------------------------------------------------------------------
echo ""
echo "========================================================================"
echo "Output layout:"
find "$OUTPUT" -name "shard_*.npz" | awk -F/ '{print $(NF-2)"/"$(NF-1)}' \
    | sort | uniq -c | awk '{printf "  %6d shards  %s\n", $1, $2}'
echo ""
echo "Vocab files:"
ls -lh "$OUTPUT"/*.pkl 2>/dev/null || echo "  (none found)"
echo "========================================================================"

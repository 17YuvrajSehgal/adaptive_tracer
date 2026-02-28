#!/bin/bash
# =============================================================================
# LMAT — Dataset Preprocessing (CPU-only job)
#
# Run this BEFORE any training job.  It reads raw LTTng CTF traces via
# babeltrace2 and writes NPZ shards that the training code loads without
# any text-parsing overhead.
#
# Estimated wall time:
#   - Small traces   (<500k requests / split) : ~30 min
#   - Large traces  (~2-5M requests / split)  : 1–2 h
# Adjust --time accordingly.
# =============================================================================

#SBATCH --account=def-naser2
#SBATCH --job-name=lmat_preprocess

# --- Resources (no GPU needed) ----------------------------------------------
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24      # 1 core for training split + up to 10 OOD workers
#SBATCH --mem=64G               # babeltrace2 can buffer several GB per split
#SBATCH --time=02:00:00

# --- Output -----------------------------------------------------------------
#SBATCH --output=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer/logs/%x-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
set -euo pipefail

PROJECT=/scratch/yuvraj17/adaptive_tracing_scratch/adaptive_tracer
DATA=/scratch/yuvraj17/adaptive_tracing_scratch/trace_data

cd "$PROJECT"
mkdir -p logs

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge
module load StdEnv/2023
module load python/3.11.5
# babeltrace2 (bt2 Python bindings) — load if available as a module,
# otherwise it must be installed in the venv.
module load babeltrace2 2>/dev/null || true

source "$PROJECT/.venv/bin/activate"

# Confirm babeltrace2 is importable
python -c "import bt2; print('bt2 version:', bt2.__version__)"

# ---------------------------------------------------------------------------
# How many OOD parallel workers?
# Use (cpus-per-task - 4) to leave headroom for the OS and the training split.
# Cap at the number of OOD splits (10 here).
# ---------------------------------------------------------------------------
N_OOD_SPLITS=10
N_WORKERS=$(( SLURM_CPUS_PER_TASK - 4 ))
N_WORKERS=$(( N_WORKERS < N_OOD_SPLITS ? N_WORKERS : N_OOD_SPLITS ))
N_WORKERS=$(( N_WORKERS < 1 ? 1 : N_WORKERS ))
echo "Using ${N_WORKERS} parallel OOD workers"

# ---------------------------------------------------------------------------
# Run preprocessing
# ---------------------------------------------------------------------------
srun python -u data-preprocessing/preprocess.py \
    --data_path "$DATA" \
    \
    --train_folder      "Train:train_id" \
    --valid_id_folder   "Valid ID:valid_id" \
    --test_id_folder    "Test ID:test_id" \
    \
    --valid_ood_folders \
        "Valid OOD (Connection):valid_ood_connection,\
Valid OOD (CPU):valid_ood_cpu,\
Valid OOD (IO):valid_ood_dumpio,\
Valid OOD (OPCache):valid_ood_opcache,\
Valid OOD (Socket):valid_ood_socket" \
    \
    --test_ood_folders \
        "Test OOD (Connection):test_ood_connection,\
Test OOD (CPU):test_ood_cpu,\
Test OOD (IO):test_ood_dumpio,\
Test OOD (OPCache):test_ood_opcache,\
Test OOD (Socket):test_ood_socket,\
Test OOD (SSL):test_ood_ssl" \
    \
    --shard_size 5000 \
    --num_proc   "${N_WORKERS}"

echo "========================================================================"
echo "Preprocessing complete.  NPZ shards written to:"
find "$DATA" -name "shard_*.npz" | head -5
echo "..."
echo "========================================================================"

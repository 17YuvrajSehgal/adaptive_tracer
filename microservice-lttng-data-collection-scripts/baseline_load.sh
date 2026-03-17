#!/bin/bash
# baseline_load.sh — Pure baseline: NO LTTng, NO OTel relay, NO LMAT
# Provides the instrumentation-free floor for overhead comparisons.
#
# Usage: ./baseline_load.sh <run_id> [duration_seconds]
#   e.g. ./baseline_load.sh run01 300
set -e

RUN_ID=${1:-run01}
DURATION=${2:-300}
EXPERIMENT_DIR=~/experiments/baseline/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}

mkdir -p "$EXPERIMENT_DIR"/load_logs

echo "🚀 BASELINE (no tracing, no LMAT): $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"

# ── Stop ALL tracing (idempotent — safe even if no sessions exist) ──────────
echo "🔇 Destroying any active LTTng sessions..."
lttng destroy --all 2>/dev/null || true
sudo lttng destroy --all 2>/dev/null || true

# ── Kill OTel relay if running ───────────────────────────────────────────────
echo "🔇 Killing OTel relay (if running)..."
pkill -f otel-to-lttng.py 2>/dev/null || true
sleep 1   # give it a moment to die

# ── Confirm nothing is tracing ───────────────────────────────────────────────
ACTIVE=$(lttng list 2>/dev/null | grep -c "Recording session" || true)
SUDO_ACTIVE=$(sudo lttng list 2>/dev/null | grep -c "Recording session" || true)
if [[ "$ACTIVE" -gt 0 ]] || [[ "$SUDO_ACTIVE" -gt 0 ]]; then
    echo "⚠️  WARNING: LTTng sessions still active. Aborting." >&2
    exit 1
fi
echo "✅ LTTng is silent. Starting pure baseline run."

RUN_START_EPOCH=$(date -u +%s)

# ── Load generator only ──────────────────────────────────────────────────────
python3 ~/load_generator.py \
    --host "$FRONTEND_HOST" \
    --users "$LOAD_USERS" \
    --duration "$DURATION" \
    --think-min 0.2 \
    --think-max 1.0 \
    --log-level WARNING \
    --output "$EXPERIMENT_DIR/load_results.csv"

RUN_END_EPOCH=$(date -u +%s)

# ── Summary ──────────────────────────────────────────────────────────────────
REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
ELAPSED=$((RUN_END_EPOCH - RUN_START_EPOCH))

cat <<EOF

✅ BASELINE $RUN_ID COMPLETE
📊 Requests    : $REQ_COUNT  (in ${ELAPSED}s)
📈 Throughput  : $(echo "scale=1; $REQ_COUNT / $ELAPSED" | bc) req/s (approx)
📁 Output      : $EXPERIMENT_DIR

EOF

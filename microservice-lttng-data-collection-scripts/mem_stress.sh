#!/bin/bash
set -e

RUN_ID=${1:-ultra_01}
DURATION=${2:-180}                     # 3 min (same as others)
EXPERIMENT_DIR=~/experiments/mem_stress/$RUN_ID
LOAD_USERS=${LOAD_USERS:-200}          # same load as others

# Strong memory pressure knobs
VM_WORKERS=${VM_WORKERS:-16}           # more workers
VM_BYTES=${VM_BYTES:-90%}              # aggressive but safer than 100%
VM_METHOD=${VM_METHOD:-all}

mkdir -p "$EXPERIMENT_DIR"/{metrics}
RUN_START_EPOCH=$(date -u +%s)

echo "🧠 ULTRA MEM Stress: $RUN_ID (${DURATION}s, ${LOAD_USERS} users, vm=${VM_WORKERS}, bytes=${VM_BYTES})"

# 1) Tracing
(cd ~ && ./collect_trace.sh anomaly_mem "$RUN_ID" "$DURATION") &
TRACE_PID=$!

# 2) Strong memory stress (allocate + touch + keep resident)
stress-ng \
  --vm "$VM_WORKERS" \
  --vm-bytes "$VM_BYTES" \
  --vm-method "$VM_METHOD" \
  --vm-keep \
  --page-in \
  --timeout "${DURATION}s" \
  --metrics-brief &
STRESS_PID=$!

echo "🔥 Memory pressure PID $STRESS_PID"

# 3) Load
python3 ~/load_generator.py \
  --host "${FRONTEND_HOST:-http://localhost:80}" \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min 0.1 \
  --think-max 0.3 \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$STRESS_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Cleanup
TRACE_DIR=~/traces/anomaly_mem/"$RUN_ID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

echo "⏸️  Prometheus flush..."
sleep 30

START_ISO=$(date -u -d "@$((RUN_START_EPOCH-30))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH+30))" '+%Y-%m-%dT%H:%M:%SZ')
./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"

REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)

cat << EOF

🧠 ULTRA MEM COMPLETE: $RUN_ID
📊 Requests: $REQ_COUNT
💾 $(du -sh "$EXPERIMENT_DIR" 2>/dev/null | cut -f1)

EOF

#!/bin/bash
set -e

RUN_ID=${1:-run01}
DURATION=${2:-100}
EXPERIMENT_DIR=~/experiments/anomaly_mem/$RUN_ID
LOAD_USERS=${LOAD_USERS:-200}          # same load as others
THINK_MIN=${THINK_MIN:-0.1}
THINK_MAX=${THINK_MAX:-0.3}

# Strong memory pressure knobs
VM_WORKERS=${VM_WORKERS:-16}           # more workers
VM_BYTES=${VM_BYTES:-90%}              # aggressive but safer than 100%
VM_METHOD=${VM_METHOD:-all}

mkdir -p "$EXPERIMENT_DIR"/{metrics,load_logs}
RUN_LOG="$EXPERIMENT_DIR/run.log"
exec > >(tee -a "$RUN_LOG") 2>&1

sudo -v

echo "🧠 ULTRA MEM Stress: $RUN_ID (${DURATION}s, ${LOAD_USERS} users, vm=${VM_WORKERS}, bytes=${VM_BYTES})"
sleep 20

RUN_START_EPOCH=$(date -u +%s)

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
  --log-level DEBUG \
  --metrics-brief &
STRESS_PID=$!

echo "🔥 Memory pressure PID $STRESS_PID"

# 3) Load
python3 ~/load_generator.py \
  --host "${FRONTEND_HOST:-http://localhost:80}" \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min "$THINK_MIN" \
  --think-max "$THINK_MAX" \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$STRESS_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Cleanup
TRACE_DIR=~/traces/anomaly_mem/"$RUN_ID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

echo "⏸️  Prometheus flush (10s)..."
sleep 10

START_ISO=$(date -u -d "@$((RUN_START_EPOCH-10))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH+10))" '+%Y-%m-%dT%H:%M:%SZ')

STEP=10s RATE_WINDOW=1m ./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"


# Summary
REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
OTEL_SPANS=$(babeltrace "$TRACE_DIR/ust" 2>/dev/null | grep -c "otel.spans" || echo 0)
BUSINESS_SPANS=$(babeltrace "$TRACE_DIR/ust" 2>/dev/null | grep -c -i "service=carts\|service=orders\|service=shipping\|service=queue-master" || echo 0)

cat << EOF

✅ $RUN_ID COMPLETE
📊 Requests: $REQ_COUNT
🔍 Spans: $OTEL_SPANS ($BUSINESS_SPANS business)
📈 Metrics: $(find "$EXPERIMENT_DIR/metrics" -type f | wc -l) files
💾 $(du -sh "$EXPERIMENT_DIR" 2>/dev/null | cut -f1)
💾 $(du -sh "$TRACE_DIR" 2>/dev/null | cut -f1)

EOF

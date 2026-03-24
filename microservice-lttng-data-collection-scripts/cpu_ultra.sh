#!/bin/bash
set -e

RUN_ID=${1:-ultra_01}
DURATION=${2:-180}           # 3min sustained
EXPERIMENT_DIR=~/experiments/cpu_ultra/$RUN_ID
LOAD_USERS=200               # high load

mkdir -p "$EXPERIMENT_DIR"/{metrics}
RUN_START_EPOCH=$(date -u +%s)

echo "💥 ULTRA CPU Stress: $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"

# 1) Tracing
(cd ~ && ./collect_trace.sh anomaly_cpu "$RUN_ID" "$DURATION") &
TRACE_PID=$!

# 2) ULTRA CPU: 12 cores + L3 cache thrashing
stress-ng \
  --cpu 12 \
  --cpu-method all \
  --cpu-load 100 \
  --timeout "${DURATION}s" \
  --metrics-brief &
STRESS_PID=$!

echo "🔥 12 cores @ 100% + cache thrash (PID $STRESS_PID)"

# 3) High load
python3 ~/load_generator.py \
  --host http://localhost:80 \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min 0.1 \
  --think-max 0.3 \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$STRESS_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Cleanup
TRACE_DIR=~/traces/anomaly_cpu/"$RUN_ID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

echo "⏸️  Prometheus flush..."
sleep 30

START_ISO=$(date -u -d "@$((RUN_START_EPOCH-30))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH+30))" '+%Y-%m-%dT%H:%M:%SZ')
./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"

# Summary
REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
OTEL_SPANS=$(babeltrace "$TRACE_DIR" 2>/dev/null | grep -c "otel.spans" || echo 0)

cat << EOF

💥 ULTRA CPU COMPLETE: $RUN_ID
📊 Requests: $REQ_COUNT (expect errors)
🔍 Spans: $OTEL_SPANS
📈 $(du -sh "$EXPERIMENT_DIR" 2>/dev/null | cut -f1)
EOF

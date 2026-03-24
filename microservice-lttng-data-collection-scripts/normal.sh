#!/bin/bash
set -euo pipefail

RUN_ID=${1:-run01}
DURATION=${2:-100}
EXPERIMENT_DIR=~/experiments/normal/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}

mkdir -p "$EXPERIMENT_DIR"/{metrics,load_logs}

echo "🚀 Normal: $RUN_ID (${DURATION}s, ${LOAD_USERS} users)"

# Make sure sudo is authenticated before background tracing starts
sudo -v

echo "⏳ Warmup for Prometheus/service stability (20s)..."
sleep 20

RUN_START_EPOCH=$(date -u +%s)

# Tracing
(cd ~ && ./collect_trace.sh normal "$RUN_ID" "$DURATION") &
TRACE_PID=$!

# Load
python3 ~/load_generator.py \
  --host "$FRONTEND_HOST" \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min 0.2 \
  --think-max 1.0 \
  --log-level WARNING \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Fix perms
TRACE_DIR=~/traces/normal/"$RUN_ID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

echo "⏸️  Prometheus flush (10s)..."
sleep 10

START_ISO=$(date -u -d "@$((RUN_START_EPOCH-10))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH+10))" '+%Y-%m-%dT%H:%M:%SZ')

STEP=10s RATE_WINDOW=1m ./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"

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
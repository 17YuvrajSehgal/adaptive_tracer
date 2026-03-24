#!/bin/bash
set -e

RUN_ID=${1:-run01}
DURATION=${2:-100}
EXPERIMENT_DIR=~/experiments/disk_ultra/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}

# Disk stress knobs (safer defaults than "10TB writes")
DISK_WORKERS=${DISK_WORKERS:-300}   # more concurrency
DISK_BYTES=${DISK_BYTES:-4G}        # per-worker target (total is still very large; tune carefully)
HDD_OPTS=${HDD_OPTS:-direct,fsync}  # stronger latency impact (fsync hurts)

mkdir -p "$EXPERIMENT_DIR"/{metrics}

echo "💥 ULTRA DISK Stress: $RUN_ID (${DURATION}s, ${LOAD_USERS} users + ${DISK_WORKERS} workers, ${DISK_BYTES}/worker, opts=${HDD_OPTS})"

sudo -v
echo "⏳ Warmup for Prometheus/service stability (20s)..."
sleep 20

RUN_START_EPOCH=$(date -u +%s)

# 1) Tracing
(cd ~ && ./collect_trace.sh anomaly_disk "$RUN_ID" "$DURATION") &
TRACE_PID=$!

# 2) Stronger DISK: more workers + fsync pressure, capped bytes
stress-ng \
  --hdd "$DISK_WORKERS" \
  --hdd-bytes "$DISK_BYTES" \
  --hdd-opts "$HDD_OPTS" \
  --timeout "${DURATION}s" \
  --metrics-brief &
STRESS_PID=$!

echo "💾 stress-ng PID $STRESS_PID"

# 3) Load (more aggressive think time)
python3 ~/load_generator.py \
  --host "${FRONTEND_HOST:-http://localhost:80}" \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min 0.05 \
  --think-max 0.2 \
  --log-level WARNING \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$STRESS_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Cleanup
TRACE_DIR=~/traces/anomaly_disk/"$RUN_ID"
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

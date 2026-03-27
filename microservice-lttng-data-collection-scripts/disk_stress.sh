#!/bin/bash
set -e

# Usage: ./disk_stress.sh <run_id> [duration_seconds]

RUN_ID=${1:-run_01}
DURATION=${2:-120}                   # default 2 minutes
EXPERIMENT_DIR=~/experiments/disk_stress/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-100}        # default 100 users
DISK_STRESS=${DISK_STRESS:-4}        # 4 disk workers

mkdir -p "$EXPERIMENT_DIR"/{metrics,load_logs}

echo "💾 Disk I/O Stress: $RUN_ID (${DURATION}s, ${LOAD_USERS} users + ${DISK_STRESS}x disk stress)"

# 1) Record precise start/end
RUN_START_EPOCH=$(date -u +%s)

# 2) Start tracing
(
  cd ~
  ./collect_trace.sh anomaly_disk "$RUN_ID" "$DURATION"
) &
TRACE_PID=$!

# 3) Start disk I/O stress (4GB writes, 4 workers)
stress-ng --io "$DISK_STRESS" \
  --hdd "$DISK_STRESS" \
  --hdd-bytes 4G \
  --timeout "$DURATION"s \
  --metrics-brief &
STRESS_PID=$!

echo "💾 Disk stress started (PID $STRESS_PID)"

# 4) Start load generator
python3 ~/load_generator.py \
  --host "$FRONTEND_HOST" \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min 0.2 \
  --think-max 1.0 \
  --log-level INFO \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

echo "⏳ Tracing ($TRACE_PID) + Disk ($STRESS_PID) + Load ($LOAD_PID)..."

# 5) Wait for all
wait "$TRACE_PID" "$STRESS_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# 6) Fix permissions
TRACE_DIR=~/traces/anomaly_disk/"$RUN_ID"
sudo chown -R "$(whoami)":"$(whoami)" "$TRACE_DIR" 2>/dev/null || true

# 7) 30s pause for Prometheus
echo "⏸️  Pausing 30s for Prometheus flush..."
sleep 30

# 8) Metrics download (+/- 30s buffer)
START_ISO=$(date -u -d "@$((RUN_START_EPOCH - 30))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH + 30))" '+%Y-%m-%dT%H:%M:%SZ')

echo "📥 Metrics: $START_ISO → $END_ISO"
./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"

# 9) Health summary
REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)
OTEL_SPANS=$(babeltrace "$TRACE_DIR" 2>/dev/null | grep -c "otel.spans" || echo 0)
BUSINESS_SPANS=$(babeltrace "$TRACE_DIR" 2>/dev/null \
  | grep "otel.spans" \
  | grep -i -E "cart|orders|checkout" \
  | wc -l || echo 0)
METRIC_FILES=$(find "$EXPERIMENT_DIR/metrics" -type f 2>/dev/null | wc -l || echo 0)

echo
echo "✅ Disk Stress: $RUN_ID"
echo "📊 Requests: $REQ_COUNT"
echo "🔍 Spans: $OTEL_SPANS  (business: $BUSINESS_SPANS)"
echo "📈 Metrics: $METRIC_FILES files"
echo "💾 $(du -sh "$EXPERIMENT_DIR")"
echo "💾 $(du -sh "$TRACE_DIR")"

#!/bin/bash
set -e

RUN_ID=${1:-ultra_01}
DURATION=${2:-180}                  # default 3min sustained
EXPERIMENT_DIR=~/experiments/disk_ultra/$RUN_ID

# Stronger load
LOAD_USERS=${LOAD_USERS:-200}

# Disk stress knobs (safer defaults than "10TB writes")
DISK_WORKERS=${DISK_WORKERS:-300}   # more concurrency
DISK_BYTES=${DISK_BYTES:-4G}        # per-worker target (total is still very large; tune carefully)
HDD_OPTS=${HDD_OPTS:-direct,fsync}  # stronger latency impact (fsync hurts)

mkdir -p "$EXPERIMENT_DIR"/{metrics,load_logs}
RUN_START_EPOCH=$(date -u +%s)

echo "💥 ULTRA DISK Stress: $RUN_ID (${DURATION}s, ${LOAD_USERS} users + ${DISK_WORKERS} workers, ${DISK_BYTES}/worker, opts=${HDD_OPTS})"

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
  --log-level INFO \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$STRESS_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Cleanup
TRACE_DIR=~/traces/anomaly_disk/"$RUN_ID"
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

💥 ULTRA DISK COMPLETE: $RUN_ID
📊 Requests: $REQ_COUNT (expect DB/storage timeouts)
🔍 Spans: $OTEL_SPANS
📈 $(du -sh "$EXPERIMENT_DIR" 2>/dev/null | cut -f1)
💾 Disk I/O: $(grep -E "^(cpu|intr)" /proc/stat 2>/dev/null | head -n 1 || echo "Use iostat/vmstat")

EOF

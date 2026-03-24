#!/bin/bash
set -e

RUN_ID=${1:-run01}
DURATION=${2:-100}
EXPERIMENT_DIR=~/experiments/net_stress/$RUN_ID
FRONTEND_HOST=${FRONTEND_HOST:-http://localhost:80}
LOAD_USERS=${LOAD_USERS:-200}

# Your host's default NIC
NET_IFACE=${NET_IFACE:-ens4}

# Strong (but safe) impairment knobs
NET_DELAY_MS=${NET_DELAY_MS:-150}      # base latency
NET_JITTER_MS=${NET_JITTER_MS:-80}     # jitter
NET_LOSS_PCT=${NET_LOSS_PCT:-3}        # packet loss %
NET_RATE=${NET_RATE:-15mbit}           # bandwidth cap
NET_BURST=${NET_BURST:-32k}
NET_LATENCY=${NET_LATENCY:-400ms}

mkdir -p "$EXPERIMENT_DIR"/{metrics}

echo "🌐 NET Anomaly: $RUN_ID (${DURATION}s, ${LOAD_USERS} users) iface=${NET_IFACE} delay=${NET_DELAY_MS}ms±${NET_JITTER_MS}ms loss=${NET_LOSS_PCT}% rate=${NET_RATE}"

sudo -v
echo "⏳ Warmup for Prometheus/service stability (20s)..."
sleep 20

RUN_START_EPOCH=$(date -u +%s)

cleanup() {
  sudo tc qdisc del dev "$NET_IFACE" root 2>/dev/null || true
}
trap cleanup EXIT

# 1) Tracing
(cd ~ && ./collect_trace.sh anomaly_net "$RUN_ID" "$DURATION") &
TRACE_PID=$!

# 2) Apply network impairment: netem (delay/jitter/loss) + tbf (rate limit)
sudo tc qdisc add dev "$NET_IFACE" root handle 1: netem \
  delay "${NET_DELAY_MS}ms" "${NET_JITTER_MS}ms" distribution normal \
  loss "${NET_LOSS_PCT}%"

sudo tc qdisc add dev "$NET_IFACE" parent 1: handle 10: tbf \
  rate "$NET_RATE" burst "$NET_BURST" latency "$NET_LATENCY"

echo "⚠️  tc netem/tbf applied on $NET_IFACE"

# 3) Load
python3 ~/load_generator.py \
  --host "${FRONTEND_HOST:-http://localhost:80}" \
  --users "$LOAD_USERS" \
  --duration "$DURATION" \
  --think-min 0.1 \
  --think-max 0.3 \
  --log-level WARNING \
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Fix perms
TRACE_DIR=~/traces/anomaly_net/"$RUN_ID"
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


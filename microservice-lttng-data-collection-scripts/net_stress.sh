#!/bin/bash
set -e

RUN_ID=${1:-ultra_01}
DURATION=${2:-180}                     # keep 180s default
EXPERIMENT_DIR=~/experiments/net_stress/$RUN_ID
LOAD_USERS=${LOAD_USERS:-200}          # keep 200 users default

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
RUN_START_EPOCH=$(date -u +%s)

echo "🌐 NET Anomaly: $RUN_ID (${DURATION}s, ${LOAD_USERS} users) iface=${NET_IFACE} delay=${NET_DELAY_MS}ms±${NET_JITTER_MS}ms loss=${NET_LOSS_PCT}% rate=${NET_RATE}"

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
  --output "$EXPERIMENT_DIR/load_results.csv" &
LOAD_PID=$!

wait "$TRACE_PID" "$LOAD_PID"

RUN_END_EPOCH=$(date -u +%s)

# Fix perms
TRACE_DIR=~/traces/anomaly_net/"$RUN_ID"
sudo chown -R "$(whoami)" "$TRACE_DIR" 2>/dev/null || true

echo "⏸️  Prometheus flush..."
sleep 30

START_ISO=$(date -u -d "@$((RUN_START_EPOCH-30))" '+%Y-%m-%dT%H:%M:%SZ')
END_ISO=$(date -u -d "@$((RUN_END_EPOCH+30))" '+%Y-%m-%dT%H:%M:%SZ')
./download_metrics.sh "$START_ISO" "$END_ISO" "$EXPERIMENT_DIR/metrics"

REQ_COUNT=$(tail -n +2 "$EXPERIMENT_DIR/load_results.csv" 2>/dev/null | wc -l || echo 0)

cat << EOF

🌐 NET ANOMALY COMPLETE: $RUN_ID
📊 Requests: $REQ_COUNT
📈 $(du -sh "$EXPERIMENT_DIR" 2>/dev/null | cut -f1)

EOF

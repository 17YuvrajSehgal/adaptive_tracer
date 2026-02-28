#!/bin/bash
# ============================================================
#  record_traces.sh — Record strace for all 40 services
#  Usage:
#    ./record_traces.sh normal
#    ./record_traces.sh cpu-stress
#    ./record_traces.sh memory-stress
#    ./record_traces.sh io-stress
#    ./record_traces.sh pod-restart
#    ./record_traces.sh bandwidth
#    ./record_traces.sh db-load
#    ./record_traces.sh verbose-log
# ============================================================
set -e

SCENARIO="${1:-normal}"
BASE_DIR="/home/sehgaluv17/lttng-final-traces"
OUT_DIR="$BASE_DIR/$SCENARIO"
LOG_FILE="$OUT_DIR/strace.log"
META_FILE="$OUT_DIR/meta.json"
LOAD_SCRIPT="/home/sehgaluv17/generateload.sh"

# All 40 service PIDs
SERVICE_PIDS=(
    728220 728442 728406 730043 729849 729875 729918
    750653 753059 752765 752786 753148 753122 752090
    753014 753109 728356 728173 728273 729815 729986
    730071 728076 728258 731304 731290 729890 731137
    729827 730810 731318 731359 731351 752306 752746
    752751 752308 752793 753176 752069
)

mkdir -p "$OUT_DIR"
echo "============================================"
echo " SCENARIO : $SCENARIO"
echo " OUTPUT   : $OUT_DIR"
echo " SERVICES : ${#SERVICE_PIDS[@]}"
echo "============================================"

# Verify all PIDs are alive
DEAD=()
for pid in "${SERVICE_PIDS[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
        DEAD+=($pid)
    fi
done
if [ ${#DEAD[@]} -gt 0 ]; then
    echo "❌ ERROR: Dead PIDs: ${DEAD[*]}"
    echo "   Run: ps aux | grep /app/ts- | awk '{print \$2}'"
    exit 1
fi
echo "✅ All ${#SERVICE_PIDS[@]} PIDs are alive"

# Build -p args
P_ARGS=$(printf -- '-p %s ' "${SERVICE_PIDS[@]}")

# Write metadata
START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
cat > "$META_FILE" << JSON
{
  "scenario": "$SCENARIO",
  "start_time": "$START_TS",
  "log_file": "$LOG_FILE",
  "service_count": ${#SERVICE_PIDS[@]},
  "pids": [$(IFS=,; echo "${SERVICE_PIDS[*]}")]
}
JSON

echo ""
echo "▶ Starting strace on all services..."
sudo strace $P_ARGS \
    -f -tt -T \
    -o "$LOG_FILE" &
STRACE_PID=$!
echo "  strace PID: $STRACE_PID"
sleep 2  # let strace attach

echo ""
echo "▶ Running load: ./generateload.sh light"
bash "$LOAD_SCRIPT" light
LOAD_EXIT=$?

echo ""
echo "▶ Stopping strace..."
sudo kill $STRACE_PID 2>/dev/null
wait $STRACE_PID 2>/dev/null || true

END_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LINE_COUNT=$(wc -l < "$LOG_FILE")
FILE_SIZE=$(du -sh "$LOG_FILE" | cut -f1)

# Update metadata with end stats
python3 -c "
import json
with open('$META_FILE') as f:
    m = json.load(f)
m['end_time']   = '$END_TS'
m['line_count'] = $LINE_COUNT
m['file_size']  = '$FILE_SIZE'
m['load_exit']  = $LOAD_EXIT
with open('$META_FILE', 'w') as f:
    json.dump(m, f, indent=2)
"

echo ""
echo "============================================"
echo " ✅ Done — $SCENARIO"
echo "    Lines : $LINE_COUNT"
echo "    Size  : $FILE_SIZE"
echo "    File  : $LOG_FILE"
echo "============================================"

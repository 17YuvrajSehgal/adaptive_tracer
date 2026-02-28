#!/bin/bash
# ============================================================
#  record_traces.sh — Record strace for all ts-* services
#  PIDs are discovered dynamically at runtime
# ============================================================
set -e

SCENARIO="${1:-normal}"
BASE_DIR="/home/sehgaluv17/lttng-final-traces"
OUT_DIR="$BASE_DIR/$SCENARIO"
LOG_FILE="$OUT_DIR/strace.log"
META_FILE="$OUT_DIR/meta.json"
LOAD_DIR="/home/sehgaluv17/train-ticket-auto-query"
LOAD_VENV="$LOAD_DIR/venv/bin/activate"

mkdir -p "$OUT_DIR"

# ── Dynamically discover all ts-* service PIDs ──────────────
echo "▶ Discovering ts-* service PIDs..."
mapfile -t SERVICE_PIDS < <(ps aux | grep '/app/ts-' | grep -v grep | awk '{print $2}' | sort -u)

if [ ${#SERVICE_PIDS[@]} -eq 0 ]; then
    echo "❌ ERROR: No ts-* service processes found."
    echo "   Are the pods running? Check: kubectl get pods"
    exit 1
fi

# Build PID -> service name map for metadata
declare -A PID_TO_SERVICE
while IFS= read -r line; do
    pid=$(echo "$line" | awk '{print $2}')
    # Extract service name from cmdline
    svc=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ' | \
          grep -oP 'ts-[a-z-]+(?=\.jar)' | head -1)
    [ -n "$svc" ] && PID_TO_SERVICE[$pid]="$svc"
done < <(ps aux | grep '/app/ts-' | grep -v grep)

echo "============================================"
echo " SCENARIO : $SCENARIO"
echo " OUTPUT   : $OUT_DIR"
echo " SERVICES : ${#SERVICE_PIDS[@]}"
echo "============================================"
echo " PIDs found:"
for pid in "${SERVICE_PIDS[@]}"; do
    svc="${PID_TO_SERVICE[$pid]:-unknown}"
    echo "   $pid  $svc"
done
echo "============================================"

# Build -p args
P_ARGS=$(printf -- '-p %s ' "${SERVICE_PIDS[@]}")

# Write metadata
START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
PID_JSON=$(printf '"%s",' "${SERVICE_PIDS[@]}" | sed 's/,$//')
python3 -c "
import json
m = {
    'scenario':      '$SCENARIO',
    'start_time':    '$START_TS',
    'log_file':      '$LOG_FILE',
    'service_count': ${#SERVICE_PIDS[@]},
    'pids': [${PID_JSON}],
    'pid_map': $(python3 -c "
import json
d = {}
$(for pid in "${!PID_TO_SERVICE[@]}"; do echo "d['$pid']='${PID_TO_SERVICE[$pid]}';"; done)
print(json.dumps(d))
")
}
with open('$META_FILE', 'w') as f:
    json.dump(m, f, indent=2)
"

echo ""
echo "▶ Starting strace on ${#SERVICE_PIDS[@]} services..."
sudo strace $P_ARGS \
    -f -tt -T \
    -o "$LOG_FILE" &
STRACE_PID=$!
echo "  strace PID: $STRACE_PID"
sleep 2

echo ""
echo "▶ Running load: generateload.sh light"
cd "$LOAD_DIR" && source "$LOAD_VENV" && bash generateload.sh $([[ "$SCENARIO" == "normal" ]] && echo minimal || echo light)
LOAD_EXIT=$?

echo ""
echo "▶ Stopping strace..."
sudo kill $STRACE_PID 2>/dev/null
wait $STRACE_PID 2>/dev/null || true

END_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LINE_COUNT=$(wc -l < "$LOG_FILE")
FILE_SIZE=$(du -sh "$LOG_FILE" | cut -f1)

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

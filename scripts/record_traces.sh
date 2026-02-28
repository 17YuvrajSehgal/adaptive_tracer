#!/bin/bash
# ============================================================
# record_traces.sh — Record strace for all ts-* services
#
# The kind control-plane container has its own PID namespace,
# so strace must run INSIDE the container. We:
#   1. Discover ts-*.jar PIDs via `crictl inspect` inside the node
#   2. Launch strace inside the container via `docker exec`
#   3. Run load from the host
#   4. Stop strace and copy the log to the host output dir
# ============================================================
set -euo pipefail

SCENARIO="${1:-normal}"
BASE_DIR="/home/sehgaluv17/lttng-final-traces"
OUT_DIR="$BASE_DIR/$SCENARIO"
LOG_FILE="$OUT_DIR/strace.log"
META_FILE="$OUT_DIR/meta.json"
LOAD_DIR="/home/sehgaluv17/train-ticket-auto-query"
LOAD_VENV="$LOAD_DIR/venv/bin/activate"
NAMESPACE="${TS_NAMESPACE:-ts}"

# The kind control-plane container name
KIND_CONTAINER="train-ticket-control-plane"
# Where strace writes its log inside the container
CONTAINER_LOG="/tmp/strace_${SCENARIO}.log"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$OUT_DIR"

# ── Discover ts-* PIDs inside the kind container ────────────────────────────
log "▶ Discovering ts-* service PIDs inside kind container..."

# One docker exec: Python reads crictl ps JSON for ts-* names+IDs, then calls
# crictl inspect per container for the host PID. Outputs "pid=name" lines.
# Uses '"'"' quoting so Python sees real " — no heredoc, no nested docker exec.
declare -A PID_TO_SERVICE
SERVICE_PIDS=()

while IFS='=' read -r _pid _svc; do
    [[ "$_pid" =~ ^[0-9]+$ ]] || continue
    SERVICE_PIDS+=("$_pid")
    PID_TO_SERVICE["$_pid"]="$_svc"
done < <(
    docker exec "$KIND_CONTAINER" bash -c 'crictl ps --output json 2>/dev/null | python3 -c '"'"'
import json, sys, subprocess
data = json.load(sys.stdin)
for c in data.get("containers", []):
    name = c.get("metadata", {}).get("name", "")
    if not name.startswith("ts-"):
        continue
    cid = c["id"]
    try:
        out = subprocess.check_output(
            ["crictl", "inspect", "--output", "json", cid],
            stderr=subprocess.DEVNULL)
        d = json.loads(out)
        pid = d.get("info", {}).get("pid", "") or d.get("status", {}).get("pid", "")
        if pid:
            print(f"{pid}={name}")
    except Exception:
        pass
'"'"' 2>/dev/null'
)

# Fallback: /proc cmdline scan (gets PIDs; strips version suffix for name)
if [ ${#SERVICE_PIDS[@]} -eq 0 ]; then
    log "   crictl method found 0 PIDs — falling back to /proc scan inside container..."
    while read -r _pid; do
        [[ "$_pid" =~ ^[0-9]+$ ]] || continue
        SERVICE_PIDS+=("$_pid")
        _svc=$(docker exec "$KIND_CONTAINER" \
            grep -oP '(?<=/)(ts-[a-z-]+)(?=-[0-9])' /proc/"$_pid"/cmdline \
            2>/dev/null | head -1 || true)
        PID_TO_SERVICE["$_pid"]="${_svc:-unknown}"
    done < <(
        docker exec "$KIND_CONTAINER" bash -c '
            for f in /proc/[0-9]*/cmdline; do
                pid="${f%/cmdline}"; pid="${pid##*/proc/}"
                cmdline=$(tr "\0" " " < "$f" 2>/dev/null || true)
                echo "$cmdline" | grep -qP "ts-[a-z].*\.jar" && echo "$pid"
            done
        ' | sort -u
    )
fi

if [ ${#SERVICE_PIDS[@]} -eq 0 ]; then
    log "❌ ERROR: No ts-* service processes found inside the kind container."
    log "   Run: docker exec $KIND_CONTAINER crictl ps | grep ts-"
    exit 1
fi



log "============================================"
log " SCENARIO : $SCENARIO"
log " OUTPUT   : $OUT_DIR"
log " SERVICES : ${#SERVICE_PIDS[@]}"
log "============================================"
for pid in "${SERVICE_PIDS[@]}"; do
    printf "   %6s  %s\n" "$pid" "${PID_TO_SERVICE[$pid]:-unknown}"
done
log "============================================"

# ── Write initial metadata ───────────────────────────────────────────────────
START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
PID_JSON=$(printf '"%s",' "${SERVICE_PIDS[@]}" | sed 's/,$//')
PID_MAP_PY=""
for pid in "${!PID_TO_SERVICE[@]}"; do
    PID_MAP_PY+="d['$pid']='${PID_TO_SERVICE[$pid]}'; "
done

python3 - <<PYEOF
import json
d = {}
${PID_MAP_PY}
m = {
    'scenario':      '$SCENARIO',
    'start_time':    '$START_TS',
    'log_file':      '$LOG_FILE',
    'service_count': ${#SERVICE_PIDS[@]},
    'pids':          [${PID_JSON}],
    'pid_map':       d,
}
with open('$META_FILE', 'w') as f:
    json.dump(m, f, indent=2)
PYEOF

# ── Start strace INSIDE the kind container ─────────────────────────────────
P_ARGS=$(printf -- '-p %s ' "${SERVICE_PIDS[@]}")

log "▶ Starting strace inside $KIND_CONTAINER on ${#SERVICE_PIDS[@]} services..."
# Run strace in background inside the container; output goes to CONTAINER_LOG
docker exec -d "$KIND_CONTAINER" bash -c \
    "strace $P_ARGS -f -tt -T \
     -e trace=network,read,write,futex,clone,execve \
     -o '$CONTAINER_LOG' 2>&1"

# Brief pause to let strace attach before traffic starts
sleep 3
log "   strace running inside container, log: $CONTAINER_LOG"

# ── Run load from the host ────────────────────────────────────────────────────
LOAD_MODE="minimal"
log "▶ Running load: generateload.sh $LOAD_MODE"
cd "$LOAD_DIR"
# shellcheck disable=SC1090
source "$LOAD_VENV"
bash generateload.sh "$LOAD_MODE"
LOAD_EXIT=$?

# ── Stop strace inside the container ────────────────────────────────────
log "▶ Stopping strace inside container..."
docker exec "$KIND_CONTAINER" bash -c \
    "pkill -f 'strace.*$CONTAINER_LOG' 2>/dev/null || pkill -x strace 2>/dev/null || true"
sleep 1

# ── Copy log from container to host ─────────────────────────────────────────
log "▶ Copying strace log from container to host..."
docker cp "$KIND_CONTAINER:$CONTAINER_LOG" "$LOG_FILE"
# Clean up inside the container
docker exec "$KIND_CONTAINER" rm -f "$CONTAINER_LOG" 2>/dev/null || true

# ── Update metadata with final stats ─────────────────────────────────────────
END_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LINE_COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
FILE_SIZE=$(du -sh "$LOG_FILE" 2>/dev/null | cut -f1 || echo "0")

python3 - <<PYEOF
import json
with open('$META_FILE') as f:
    m = json.load(f)
m['end_time']   = '$END_TS'
m['line_count'] = $LINE_COUNT
m['file_size']  = '$FILE_SIZE'
m['load_exit']  = $LOAD_EXIT
with open('$META_FILE', 'w') as f:
    json.dump(m, f, indent=2)
PYEOF

log ""
log "============================================"
log " ✅ Done — $SCENARIO"
log "    Lines : $LINE_COUNT"
log "    Size  : $FILE_SIZE"
log "    File  : $LOG_FILE"
log "============================================"

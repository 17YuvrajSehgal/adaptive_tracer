#!/bin/bash
# ============================================================
# record_traces.sh — Record strace for all ts-* services
#
# The kind control-plane container has its own PID namespace,
# so strace must run INSIDE the container. We:
#   1. Discover ts-*.jar PIDs via crictl inside the node
#   2. Launch strace inside the container via `docker exec`
#   3. Run load from the host (output shown live)
#   4. Stop strace, copy the log to the host output dir
#
# Usage:
#   ./record_traces.sh [scenario] [load_mode]
#   LOAD_MODE=normal ./record_traces.sh cpu-stress
# ============================================================
set -uo pipefail   # NOTE: -e removed so load failures don't skip the copy step

SCENARIO="${1:-normal}"
LOAD_MODE="${LOAD_MODE:-${2:-minimal}}"
BASE_DIR="/home/sehgaluv17/lttng-final-traces"
OUT_DIR="$BASE_DIR/$SCENARIO"
LOG_FILE="$OUT_DIR/strace.log"
META_FILE="$OUT_DIR/meta.json"
LIVE_LOG="$OUT_DIR/run.log"          # tee target: full session log on disk
LOAD_DIR="/home/sehgaluv17/train-ticket-auto-query"
LOAD_VENV="$LOAD_DIR/venv/bin/activate"

# The kind control-plane container name
KIND_CONTAINER="train-ticket-control-plane"

# ── Logging helpers ───────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LIVE_LOG"; }
logn() { printf "[$(date '+%H:%M:%S')] %s" "$*" | tee -a "$LIVE_LOG"; }  # no newline

mkdir -p "$OUT_DIR"
: > "$LIVE_LOG"   # truncate/create run log

log "============================================================"
log " record_traces.sh starting"
log "   SCENARIO  : $SCENARIO"
log "   LOAD_MODE : $LOAD_MODE"
log "   OUTPUT    : $OUT_DIR"
log "============================================================"

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
        _svc=$(docker exec "$KIND_CONTAINER" bash -c \
            "tr '\0' ' ' < /proc/$_pid/cmdline 2>/dev/null \
             | grep -oP '(?<=/)(ts-[a-z-]+)(?=-[0-9])' | head -1" 2>/dev/null || true)
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
log " SERVICES : ${#SERVICE_PIDS[@]}"
log "============================================"
for pid in "${SERVICE_PIDS[@]}"; do
    printf "   %6s  %s\n" "$pid" "${PID_TO_SERVICE[$pid]:-unknown}" | tee -a "$LIVE_LOG"
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
    'load_mode':     '$LOAD_MODE',
    'start_time':    '$START_TS',
    'log_file':      '$LOG_FILE',
    'service_count': ${#SERVICE_PIDS[@]},
    'pids':          [${PID_JSON}],
    'pid_map':       d,
}
with open('$META_FILE', 'w') as f:
    json.dump(m, f, indent=2)
PYEOF

# ── Start strace, piping output directly to the host log file ────────────────
# Key insight: running docker exec in the BACKGROUND on the host (not -d) lets us
# pipe strace's stdout straight to $LOG_FILE.  No temp file inside the container,
# no docker cp, no file-disappears-after-pkill mystery.
P_ARGS=$(printf -- '-p %s ' "${SERVICE_PIDS[@]}")

log "▶ Starting strace inside $KIND_CONTAINER on ${#SERVICE_PIDS[@]} services..."
log "   syscalls : network,read,write,futex,clone,execve"
log "   host log : $LOG_FILE  (written live via docker exec stdout)"

: > "$LOG_FILE"   # ensure the file exists immediately so tail -f can open it

docker exec "$KIND_CONTAINER" bash -c \
    "strace $P_ARGS -f -tt -T \
     -e trace=network,read,write,futex,clone,execve \
     2>&1" \
    >> "$LOG_FILE" &
STRACE_BG_PID=$!

# Brief pause to let strace attach before confirming
sleep 3

# Verify strace is actually running inside the container
STRACE_INNER_PID=$(docker exec "$KIND_CONTAINER" \
    pgrep -x strace 2>/dev/null | head -1 || true)
if [ -n "$STRACE_INNER_PID" ]; then
    log "   strace PID inside container: $STRACE_INNER_PID ✅"
else
    log "   ⚠️  strace process not found inside container — it may have failed to attach"
    log "   Hint: try: docker exec $KIND_CONTAINER strace --version"
fi

# ── Live tail of strace log in background (prints to terminal) ───────────────
log "▶ Live strace output (sampling every 5s from $LOG_FILE):"
(
    sleep 5
    while true; do
        COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "?")
        SAMPLE=$(tail -3 "$LOG_FILE" 2>/dev/null || true)
        echo "[$(date '+%H:%M:%S')] [strace] lines so far: $COUNT" | tee -a "$LIVE_LOG"
        if [ -n "$SAMPLE" ]; then
            echo "$SAMPLE" | sed 's/^/   /' | tee -a "$LIVE_LOG"
        fi
        sleep 5
    done
) &
TAIL_PID=$!

# ── Run load from the host ────────────────────────────────────────────────────
log "▶ Running load: generateload.sh $LOAD_MODE"
log "   (load output streamed live below)"
log "------------------------------------------------------------"

cd "$LOAD_DIR"
# shellcheck disable=SC1090
source "$LOAD_VENV"
# Pipe through tee so output appears live AND lands in run.log.
# PIPESTATUS[0] gives the real generateload.sh exit code (not tee's).
bash generateload.sh "$LOAD_MODE" 2>&1 | tee -a "$LIVE_LOG"
LOAD_EXIT=${PIPESTATUS[0]}

log "------------------------------------------------------------"
log "   generateload.sh exited with code $LOAD_EXIT"

# ── Stop the live tail ────────────────────────────────────────────────────────
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true

# ── Stop strace inside the container then wait for docker exec to finish ─────
log "▶ Stopping strace inside container..."
# Send SIGTERM to the strace process inside the container.
# strace will detach from all traced PIDs and flush its output buffer.
docker exec "$KIND_CONTAINER" \
    pkill -x strace 2>/dev/null || true

# Wait for the background docker exec to finish (ensures LOG_FILE is fully written).
wait "$STRACE_BG_PID" 2>/dev/null || true
log "   docker exec finished — log should be fully flushed"


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
if [ "$LINE_COUNT" -gt 0 ]; then
    log " ✅ Done — $SCENARIO"
else
    log " ⚠️  Done (EMPTY LOG) — $SCENARIO"
fi
log "    Lines     : $LINE_COUNT"
log "    Size      : $FILE_SIZE"
log "    strace log: $LOG_FILE"
log "    run log   : $LIVE_LOG"
log "    meta      : $META_FILE"
log "    load exit : $LOAD_EXIT"
log "============================================"

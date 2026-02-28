#!/bin/bash
# ============================================================
# record_traces.sh — Record strace for all ts-* services
# PIDs are discovered via the kind control-plane container.
#
# FIX: crictl inspect returns the PID *inside* the container
# namespace (usually 1 or a small number). strace on the host
# needs the HOST-namespace PID. We translate using NStgid from
# /proc/<pid>/status which Linux always exposes on the host,
# even for processes inside nested PID namespaces.
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

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p "$OUT_DIR"

# ── Discover ts-* service PIDs ────────────────────────────────────────────────
# Strategy:
#   1. Use `docker exec` into the kind node + `crictl ps` to list ts-* containers.
#   2. For each container get its *init PID inside the kind node's PID namespace*
#      via `crictl inspect .info.pid`.
#   3. That PID is inside the kind container's PID namespace, NOT the host's.
#      Translate to host PID by reading NStgid from the kind container's /proc:
#        docker exec kind-node cat /proc/<container-init-pid>/status
#      NStgid lists PIDs from innermost to outermost namespace; the LAST value
#      is the host-visible PID that strace can attach to.
#   4. Fallback: scan /proc on the host directly for java ts-*.jar cmdlines
#      (works if kind shares the host PID namespace, which it does by default).
log "▶ Discovering ts-* service PIDs via kind container..."

mapfile -t SERVICE_PIDS < <(
  docker exec "$KIND_CONTAINER" bash -c '
    crictl ps --output json 2>/dev/null \
    | python3 -c "
import json, sys
data = json.load(sys.stdin)
for c in data.get(\"containers\", []):
    name = c.get(\"metadata\", {}).get(\"name\", \"\")
    if name.startswith(\"ts-\"):
        print(c[\"id\"])
"
  ' | while read -r cid; do
      # Get the PID of the container init process as seen inside the kind node
      container_pid=$(docker exec "$KIND_CONTAINER" bash -c \
        "crictl inspect --output json '$cid' 2>/dev/null \
         | python3 -c \"import json,sys; d=json.load(sys.stdin); print(d.get('info',{}).get('pid',''))\"" \
        2>/dev/null)

      [[ -z "$container_pid" || ! "$container_pid" =~ ^[0-9]+$ ]] && continue

      # Translate: read NStgid from /proc inside the kind node.
      # NStgid line looks like: "NStgid:  1   12345"
      # The LAST field is the outermost (host) PID.
      host_pid=$(docker exec "$KIND_CONTAINER" bash -c \
        "awk '/^NStgid/{print \$NF}' /proc/$container_pid/status 2>/dev/null" \
        2>/dev/null)

      [[ "$host_pid" =~ ^[0-9]+$ ]] && echo "$host_pid"
  done | grep -E '^[0-9]+$' | sort -u
)

# ── Fallback: scan /proc on the host directly ─────────────────────────────────
if [ ${#SERVICE_PIDS[@]} -eq 0 ]; then
    log "   crictl/NStgid method found 0 PIDs — falling back to host /proc scan..."
    mapfile -t SERVICE_PIDS < <(
        for cmdfile in /proc/[0-9]*/cmdline; do
            pid_num="${cmdfile%/cmdline}"
            pid_num="${pid_num##*/proc/}"
            cmdline=$(tr '\0' ' ' < "$cmdfile" 2>/dev/null || true)
            if echo "$cmdline" | grep -qP 'ts-[a-z0-9-]+\.jar'; then
                echo "$pid_num"
            fi
        done | sort -u
    )
fi

if [ ${#SERVICE_PIDS[@]} -eq 0 ]; then
    log "❌ ERROR: No ts-* service processes found."
    log "   Check: kubectl get pods -n $NAMESPACE"
    log "   Check: docker exec $KIND_CONTAINER crictl ps | grep ts-"
    exit 1
fi

# ── Verify PIDs are actually alive on the host ────────────────────────────────
VALID_PIDS=()
for pid in "${SERVICE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
        VALID_PIDS+=("$pid")
    else
        log "   ⚠  PID $pid not visible on host — skipping"
    fi
done

if [ ${#VALID_PIDS[@]} -eq 0 ]; then
    log "❌ ERROR: All discovered PIDs are invalid from the host perspective."
    log "   This usually means the kind container uses a separate PID namespace."
    log "   Try running strace INSIDE the kind container instead:"
    log "     docker exec -it $KIND_CONTAINER bash"
    log "     strace -f -p <pid-inside-node> -o /tmp/strace.log"
    exit 1
fi
SERVICE_PIDS=("${VALID_PIDS[@]}")

# ── Build PID → service name map ─────────────────────────────────────────────
declare -A PID_TO_SERVICE
for pid in "${SERVICE_PIDS[@]}"; do
    svc=$(cat "/proc/$pid/cmdline" 2>/dev/null \
          | tr '\0' ' ' \
          | grep -oP 'ts-[a-z0-9-]+(?=\.jar)' \
          | head -1 || true)
    [ -n "$svc" ] && PID_TO_SERVICE[$pid]="$svc" || PID_TO_SERVICE[$pid]="unknown"
done

log "============================================"
log " SCENARIO : $SCENARIO"
log " OUTPUT   : $OUT_DIR"
log " SERVICES : ${#SERVICE_PIDS[@]}"
log "============================================"
for pid in "${SERVICE_PIDS[@]}"; do
    printf "   %6s  %s\n" "$pid" "${PID_TO_SERVICE[$pid]}"
done
log "============================================"

# ── Build strace -p args ──────────────────────────────────────────────────────
P_ARGS=$(printf -- '-p %s ' "${SERVICE_PIDS[@]}")

# ── Write initial metadata ────────────────────────────────────────────────────
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

# ── Start strace ──────────────────────────────────────────────────────────────
log "▶ Starting strace on ${#SERVICE_PIDS[@]} services..."
sudo strace $P_ARGS \
    -f \
    -tt \
    -T \
    -e trace=network,read,write,futex,clone,execve \
    -o "$LOG_FILE" &
STRACE_PID=$!
log "   strace PID: $STRACE_PID"
sleep 2   # let strace attach before traffic starts

# ── Run load ──────────────────────────────────────────────────────────────────
LOAD_MODE="minimal"
log "▶ Running load: generateload.sh $LOAD_MODE"
cd "$LOAD_DIR"
# shellcheck disable=SC1090
source "$LOAD_VENV"
bash generateload.sh "$LOAD_MODE"
LOAD_EXIT=$?

# ── Stop strace ───────────────────────────────────────────────────────────────
log "▶ Stopping strace..."
sudo kill "$STRACE_PID" 2>/dev/null || true
wait "$STRACE_PID" 2>/dev/null || true

# ── Update metadata with final stats ─────────────────────────────────────────
END_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LINE_COUNT=$(wc -l < "$LOG_FILE")
FILE_SIZE=$(du -sh "$LOG_FILE" | cut -f1)

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

#!/bin/bash
# ============================================================
# record_all.sh — Record ALL scenarios sequentially
# Runs: normal, cpu-stress, memory-stress, io-stress,
#       bandwidth, db-load, pod-restart, verbose-log
#
# Usage: ./record_all.sh [--force] [--load-mode <mode>]
#   --force              Re-record even if a trace already exists
#   --load-mode <mode>   Load mode passed to generateload.sh (default: minimal)
#
# All output is also written to recording.log in the current directory.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
BASE_DIR="/home/sehgaluv17/lttng-final-traces"
COOLDOWN=30   # seconds between scenarios to let JVMs recover
FORCE=false
LOAD_MODE="minimal"
SESSION_LOG="$BASE_DIR/recording.log"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ "${1:-}" != "" ]]; do
    case "$1" in
        --force)      FORCE=true; shift ;;
        --load-mode)  LOAD_MODE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
export LOAD_MODE

mkdir -p "$BASE_DIR"
: > "$SESSION_LOG"

# ── Logging helper (writes to terminal + session log) ─────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SESSION_LOG"; }

log "╔══════════════════════════════════════════════╗"
log "║         record_all.sh — starting             ║"
log "╠══════════════════════════════════════════════╣"
log "║  LOAD_MODE : $LOAD_MODE"
log "║  FORCE     : $FORCE"
log "║  SESSION   : $SESSION_LOG"
log "╚══════════════════════════════════════════════╝"

SCENARIOS=(
    normal
    cpu-stress
    memory-stress
    io-stress
    bandwidth
    db-load
    pod-restart
    verbose-log
)
TOTAL=${#SCENARIOS[@]}

# ── Pre-flight skip check ─────────────────────────────────────────────────────
SKIP=()
for s in "${SCENARIOS[@]}"; do
    if [ -f "$BASE_DIR/$s/strace.log" ] && [ "$FORCE" = false ]; then
        SKIP+=("$s")
    fi
done

if [ ${#SKIP[@]} -gt 0 ]; then
    log "⚠️  Already recorded (use --force to re-record):"
    for s in "${SKIP[@]}"; do
        SIZE=$(du -sh "$BASE_DIR/$s/strace.log" 2>/dev/null | cut -f1 || echo '?')
        LINES=$(wc -l < "$BASE_DIR/$s/strace.log" 2>/dev/null || echo '?')
        log "      $s  ($SIZE, $LINES lines)"
    done
fi

# ── Helper: run one scenario ──────────────────────────────────────────────────
run_scenario() {
    local SCENARIO="$1"
    local N="$2"
    local IS_ANOMALY="${3:-false}"

    # Skip if trace exists and not forcing
    if [ -f "$BASE_DIR/$SCENARIO/strace.log" ] && [ "$FORCE" = false ]; then
        log "⏭  Skipping '$SCENARIO' — trace exists (delete or use --force)"
        return 0
    fi

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "▶ SCENARIO $N/$TOTAL — $SCENARIO  [$(date '+%H:%M:%S')]"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ "$IS_ANOMALY" = true ]; then
        log "  Injecting anomaly: $SCENARIO"
        bash "$SCRIPT_DIR/inject_anomaly.sh" start "$SCENARIO" 2>&1 | tee -a "$SESSION_LOG"
        log "  Waiting 5s for anomaly to stabilise..."
        sleep 5
    fi

    # Record — do not abort the whole run if one scenario fails
    bash "$SCRIPT_DIR/record_traces.sh" "$SCENARIO" 2>&1 | tee -a "$SESSION_LOG" || {
        log "  ⚠️  record_traces.sh exited non-zero for '$SCENARIO' — continuing"
    }

    if [ "$IS_ANOMALY" = true ]; then
        log "  Stopping anomaly..."
        bash "$SCRIPT_DIR/inject_anomaly.sh" stop 2>&1 | tee -a "$SESSION_LOG"
    fi

    # Quick result check
    LOG="$BASE_DIR/$SCENARIO/strace.log"
    if [ -f "$LOG" ]; then
        SIZE=$(du -sh "$LOG" | cut -f1)
        LINES=$(wc -l < "$LOG")
        log "  ✅ $SCENARIO — $LINES lines, $SIZE"
    else
        log "  ❌ $SCENARIO — strace.log MISSING"
    fi

    log "  Cooldown ${COOLDOWN}s (letting JVMs recover)..."
    sleep "$COOLDOWN"
}

# ── Run all scenarios ─────────────────────────────────────────────────────────
run_scenario "normal"        1 false
run_scenario "cpu-stress"    2 true
run_scenario "memory-stress" 3 true
run_scenario "io-stress"     4 true
run_scenario "bandwidth"     5 true
run_scenario "db-load"       6 true
run_scenario "pod-restart"   7 true
run_scenario "verbose-log"   8 true

# ── Final summary ─────────────────────────────────────────────────────────────
echo "" | tee -a "$SESSION_LOG"
echo "╔══════════════════════════════════════════════════════╗" | tee -a "$SESSION_LOG"
echo "║              RECORDING COMPLETE                      ║" | tee -a "$SESSION_LOG"
echo "╠══════════════════════════════════════════════════════╣" | tee -a "$SESSION_LOG"
for s in "${SCENARIOS[@]}"; do
    LOG="$BASE_DIR/$s/strace.log"
    if [ -f "$LOG" ]; then
        SIZE=$(du -sh "$LOG" | cut -f1)
        LINES=$(wc -l < "$LOG")
        printf "║  %-22s  %6s  %8d lines  ║\n" "$s" "$SIZE" "$LINES" | tee -a "$SESSION_LOG"
    else
        printf "║  %-22s  %-26s  ║\n" "$s" "❌ MISSING" | tee -a "$SESSION_LOG"
    fi
done
echo "╠══════════════════════════════════════════════════════╣" | tee -a "$SESSION_LOG"
printf "║  Session log: %-38s║\n" "$SESSION_LOG" | tee -a "$SESSION_LOG"
echo "╚══════════════════════════════════════════════════════╝" | tee -a "$SESSION_LOG"
echo ""
echo "Next step: python3 $SCRIPT_DIR/parse_strace.py --scenario <name>"
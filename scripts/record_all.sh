#!/bin/bash
# ============================================================
#  record_all.sh — Record ALL scenarios sequentially
#  Runs: normal, cpu-stress, memory-stress, io-stress,
#        bandwidth, db-load, pod-restart, verbose-log
#
#  Usage: ./record_all.sh
# ============================================================
set -e

BASE_DIR="/home/sehgaluv17/lttng-final-traces"
COOLDOWN=30   # seconds between scenarios

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_normal() {
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "SCENARIO 1/8 — normal"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    bash record_traces.sh normal
}

run_anomaly() {
    SCENARIO="$1"
    N="$2"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "SCENARIO $N/8 — $SCENARIO"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  Injecting anomaly: $SCENARIO"
    bash inject_anomaly.sh start "$SCENARIO"
    sleep 5   # let anomaly settle before strace starts
    bash record_traces.sh "$SCENARIO"
    log "  Stopping anomaly..."
    bash inject_anomaly.sh stop
    log "  Cooldown ${COOLDOWN}s..."
    sleep $COOLDOWN
}

# Ensure output dirs are clear of old partial runs
for SCENARIO in normal cpu-stress memory-stress io-stress \
                bandwidth db-load pod-restart verbose-log; do
    if [ -f "$BASE_DIR/$SCENARIO/strace.log" ]; then
        log "⚠️  Existing trace found for '$SCENARIO' — skipping (delete to re-record)"
    fi
done

run_normal

sleep $COOLDOWN

run_anomaly "cpu-stress"    2
run_anomaly "memory-stress" 3
run_anomaly "io-stress"     4
run_anomaly "bandwidth"     5
run_anomaly "db-load"       6
run_anomaly "pod-restart"   7
run_anomaly "verbose-log"   8

# Final summary
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║         RECORDING COMPLETE               ║"
echo "╠══════════════════════════════════════════╣"
for SCENARIO in normal cpu-stress memory-stress io-stress \
                bandwidth db-load pod-restart verbose-log; do
    LOG="$BASE_DIR/$SCENARIO/strace.log"
    if [ -f "$LOG" ]; then
        SIZE=$(du -sh "$LOG" | cut -f1)
        LINES=$(wc -l < "$LOG")
        printf "║  %-20s %8s  %8d lines ║\n" "$SCENARIO" "$SIZE" "$LINES"
    else
        printf "║  %-20s  %-20s ║\n" "$SCENARIO" "MISSING"
    fi
done
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Next step: python3 scripts/parse_strace.py --scenario <name>"

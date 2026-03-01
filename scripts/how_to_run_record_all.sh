#!/bin/bash
# ============================================================
# how_to_run_record_all.sh — Quick reference for recording
# ============================================================

# Option A — Record all scenarios at once (~2-3 hours)
#   All output goes to /home/sehgaluv17/lttng-final-traces/recording.log
#   AND streams live to your terminal.
cd ~/adaptive_tracer/scripts
./record_all.sh

# Option A with force re-record:
./record_all.sh --force

# Option A with a different load mode (default: minimal):
./record_all.sh --load-mode normal

# ──────────────────────────────────────────────────────────────
# Option B — Record one scenario at a time
# ──────────────────────────────────────────────────────────────

# Normal (no anomaly):
./record_traces.sh normal

# Anomaly scenarios — inject BEFORE recording, stop AFTER:
bash inject_anomaly.sh start cpu-stress
./record_traces.sh cpu-stress
bash inject_anomaly.sh stop

bash inject_anomaly.sh start memory-stress
./record_traces.sh memory-stress
bash inject_anomaly.sh stop

bash inject_anomaly.sh start io-stress
./record_traces.sh io-stress
bash inject_anomaly.sh stop

bash inject_anomaly.sh start bandwidth
./record_traces.sh bandwidth
bash inject_anomaly.sh stop

bash inject_anomaly.sh start db-load
./record_traces.sh db-load
bash inject_anomaly.sh stop

bash inject_anomaly.sh start pod-restart
./record_traces.sh pod-restart
bash inject_anomaly.sh stop

bash inject_anomaly.sh start verbose-log
./record_traces.sh verbose-log
bash inject_anomaly.sh stop

# ──────────────────────────────────────────────────────────────
# What to look for while recording
# ──────────────────────────────────────────────────────────────
# record_traces.sh prints live status:
#   [HH:MM:SS] [strace] lines so far: NNNN   ← growing = strace capturing
#   [HH:MM:SS] ✅ Done — normal             ← success
#   [HH:MM:SS] ⚠️  Done (EMPTY LOG)          ← strace attached but no traffic
#
# To manually check strace is writing inside the container:
#   docker exec train-ticket-control-plane wc -l /tmp/strace_normal.log
#
# Per-scenario output:
#   /home/sehgaluv17/lttng-final-traces/<scenario>/strace.log   — raw strace
#   /home/sehgaluv17/lttng-final-traces/<scenario>/run.log      — full session log
#   /home/sehgaluv17/lttng-final-traces/<scenario>/meta.json    — metadata

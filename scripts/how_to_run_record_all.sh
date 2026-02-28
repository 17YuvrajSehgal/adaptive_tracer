# Option A — record everything at once (recommended, ~2-3 hours)
./record_all.sh 2>&1 | tee recording.log

# Option B — record one scenario at a time
./record_traces.sh normal
bash inject_anomaly.sh start cpu-stress && ./record_traces.sh cpu-stress && bash inject_anomaly.sh stop
bash inject_anomaly.sh start memory-stress && ./record_traces.sh memory-stress && bash inject_anomaly.sh stop
# ... etc

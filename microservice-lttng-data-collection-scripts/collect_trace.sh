#!/bin/bash
TYPE=$1
RUN=$2
DURATION=${3:-120}
OUTPUT_DIR=~/traces/$TYPE/$RUN

mkdir -p $OUTPUT_DIR/{kernel,ust}

# UST session (no sudo)
lttng create sockshop-ust --output=$OUTPUT_DIR/ust
lttng enable-event --python otel.spans
lttng start

# Kernel session (sudo)
sudo lttng create sockshop-kernel --output=$OUTPUT_DIR/kernel
sudo lttng enable-event -k --all '*'
sudo lttng start

# OTel relay
python3 /home/sehgaluv17/agents/otel-to-lttng.py $4 &
RELAY_PID=$!

echo "[$TYPE/$RUN] FULL TRACING (Kernel+UST) for ${DURATION}s..."

if [[ "$TYPE" == anomaly_cpu* ]]; then
    stress-ng --cpu 4 --timeout ${DURATION}s &
    STRESS_PID=$!
fi

sleep $DURATION

# Cleanup
kill $RELAY_PID 2>/dev/null && wait $RELAY_PID
[[ -n "$STRESS_PID" ]] && kill $STRESS_PID 2>/dev/null && wait $STRESS_PID
lttng stop && sudo lttng stop
lttng destroy && sudo lttng destroy

mkdir -p ~/traces
echo "$TYPE,$RUN,$(date -u),${DURATION}s,FULL" >> ~/traces/metadata.csv

echo "[$TYPE/$RUN] DONE. $(du -sh $OUTPUT_DIR)"

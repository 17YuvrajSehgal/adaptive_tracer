#!/bin/bash
# ============================================================
#  inject_anomaly.sh — Inject various anomaly conditions
#  Sourced by record_all.sh before each anomaly scenario
#  Usage: source inject_anomaly.sh <scenario>  (to set cleanup fn)
#         bash inject_anomaly.sh start <scenario>
#         bash inject_anomaly.sh stop
# ============================================================

CMD="${1:-start}"
SCENARIO="${2:-}"
CLEANUP_FILE="/tmp/anomaly_cleanup_pids"

start_cpu_stress() {
    echo "🔥 Starting CPU stress (all cores)..."
    CPU_COUNT=$(nproc)
    stress-ng --cpu "$CPU_COUNT" --cpu-load 85 --timeout 0 &
    echo $! >> "$CLEANUP_FILE"
    echo "   stress-ng PID: $!"
}

start_memory_stress() {
    echo "🧠 Starting memory pressure (6GB)..."
    stress-ng --vm 4 --vm-bytes 1500m --timeout 0 &
    echo $! >> "$CLEANUP_FILE"
    echo "   stress-ng PID: $!"
}

start_io_stress() {
    echo "💾 Starting IO stress..."
    stress-ng --io 8 --hdd 4 --timeout 0 &
    echo $! >> "$CLEANUP_FILE"
    echo "   stress-ng PID: $!"
}

start_bandwidth_throttle() {
    echo "🌐 Throttling network bandwidth to 10Mbit..."
    # Throttle on cluster interface (adjust eth0 if needed)
    IFACE=$(ip route | grep default | awk '{print $5}' | head -1)
    sudo tc qdisc add dev "$IFACE" root tbf rate 10mbit burst 32kbit latency 400ms
    echo "tc:$IFACE" >> "$CLEANUP_FILE"
    echo "   Throttled: $IFACE → 10Mbit"
}

start_db_load() {
    echo "🗄  Starting DB overload (heavy MySQL queries)..."
    # Hit MySQL repeatedly via kubectl exec
    kubectl exec -it tsdb-mysql-0 -- bash -c \
        "while true; do mysql -uroot -ppassword ts -e \
        'SELECT * FROM orders o JOIN order_items oi ON o.id=oi.order_id LIMIT 100000;' \
        2>/dev/null; done" &
    echo $! >> "$CLEANUP_FILE"
    echo "   DB load PID: $!"
}

start_pod_restart() {
    echo "🔄 Scheduling pod restarts every 30s..."
    (while true; do
        kubectl delete pod -l app=ts-order-service --grace-period=0 2>/dev/null
        sleep 30
        kubectl delete pod -l app=ts-payment-service --grace-period=0 2>/dev/null
        sleep 30
    done) &
    echo $! >> "$CLEANUP_FILE"
    echo "   Restart loop PID: $!"
}

start_verbose_log() {
    echo "📝 Enabling verbose logging on ts-order-service..."
    kubectl exec -it deploy/ts-order-service -- \
        curl -s -X POST "http://localhost:8080/actuator/loggers/ROOT" \
        -H "Content-Type: application/json" \
        -d '{"configuredLevel":"TRACE"}' || true
    echo "verbose_log" >> "$CLEANUP_FILE"
}

stop_all() {
    echo "🛑 Stopping anomaly injectors..."
    if [ ! -f "$CLEANUP_FILE" ]; then
        echo "   No cleanup file found"
        return
    fi
    while IFS= read -r entry; do
        if [[ "$entry" == tc:* ]]; then
            IFACE="${entry#tc:}"
            sudo tc qdisc del dev "$IFACE" root 2>/dev/null && \
                echo "   Removed tc qdisc on $IFACE"
        elif [[ "$entry" == "verbose_log" ]]; then
            kubectl exec -it deploy/ts-order-service -- \
                curl -s -X POST "http://localhost:8080/actuator/loggers/ROOT" \
                -H "Content-Type: application/json" \
                -d '{"configuredLevel":"INFO"}' 2>/dev/null || true
            echo "   Restored logging level to INFO"
        else
            kill "$entry" 2>/dev/null && echo "   Killed PID $entry"
        fi
    done < "$CLEANUP_FILE"
    rm -f "$CLEANUP_FILE"
    echo "✅ All anomaly processes stopped"
}

rm -f "$CLEANUP_FILE"

case "$CMD" in
    start)
        case "$SCENARIO" in
            cpu-stress)       start_cpu_stress ;;
            memory-stress)    start_memory_stress ;;
            io-stress)        start_io_stress ;;
            bandwidth)        start_bandwidth_throttle ;;
            db-load)          start_db_load ;;
            pod-restart)      start_pod_restart ;;
            verbose-log)      start_verbose_log ;;
            *) echo "Unknown scenario: $SCENARIO"; exit 1 ;;
        esac
        ;;
    stop) stop_all ;;
    *) echo "Usage: $0 start <scenario> | stop"; exit 1 ;;
esac

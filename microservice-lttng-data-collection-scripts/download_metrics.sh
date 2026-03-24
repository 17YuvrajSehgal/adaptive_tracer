#!/bin/bash
# download_metrics.sh - Full version w/ output directory
PROMETHEUS="http://34.58.215.226:9090"
STEP="30s"

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <start> <end> [output_dir]"
    echo "Examples:"
    echo "  $0 '1 hour ago' now"
    echo "  $0 '1 hour ago' now anomaly_run_01"
    echo "  $0 '2026-03-02T08:00:00Z' '2026-03-02T08:30:00Z' baseline"
    exit 1
fi

START_RAW=$1
END_RAW=$2
OUTPUT_DIR=${3:-metrics}  # Default: 'metrics'

# Convert to Unix timestamps
START=$(date -u -d "$START_RAW" +%s 2>/dev/null || { echo "❌ Invalid start: $START_RAW"; exit 1; })
END=$(date -u -d "$END_RAW" +%s 2>/dev/null || { echo "❌ Invalid end: $END_RAW"; exit 1; })

echo "📥 Downloading $START_RAW → $END_RAW"
echo "📁 Output: $OUTPUT_DIR/"
echo "⏱️  Range: $(date -u -d "@$START") → $(date -u -d "@$END")"

mkdir -p "$OUTPUT_DIR"

download() {
    local name=$1 query=$2
    local file="$OUTPUT_DIR/$name.json"
    curl -s -G "$PROMETHEUS/api/v1/query_range" \
      --data-urlencode "query=$query" \
      --data-urlencode "start=$START" \
      --data-urlencode "end=$END" \
      --data-urlencode "step=$STEP" > "$file"
    
    if [ $(wc -c < "$file") -gt 200 ]; then
        echo "✅ $name.json ($(wc -c < "$file") bytes)"
    else
        echo "⚠️  $name.json (empty/no data)"
    fi
}

# ========== VM SYSTEM ==========
echo "=== VM Metrics ==="
download "vm_cpu" "100-(avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m]))*100)"
download "vm_memory" "(1-node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes)*100"
download "vm_disk" "100-(node_filesystem_avail_bytes{job=\"node-exporter\",fstype!=\"rootfs\"}/node_filesystem_size_bytes{job=\"node-exporter\",fstype!=\"rootfs\"})*100"
download "vm_network_receive" "rate(node_network_receive_bytes_total[5m])"
download "vm_network_transmit" "rate(node_network_transmit_bytes_total[5m])"

# ========== SOCK SHOP SERVICES ==========
SERVICES="catalogue cart orders payment shipping user frontend"
echo "=== Sock Shop Services ==="
for service in $SERVICES; do
    echo "--- $service ---"
    download "${service}_qps" "sum(rate(request_duration_seconds_count{job=\"$service\"}[5m]))"
    download "${service}_p95_latency" "histogram_quantile(0.95,sum(rate(request_duration_seconds_bucket{job=\"$service\"}[5m]))by(le))"
    download "${service}_p50_latency" "histogram_quantile(0.50,sum(rate(request_duration_seconds_bucket{job=\"$service\"}[5m]))by(le))"
    download "${service}_errors" "rate(request_duration_seconds_count{job=\"$service\",status_code=~\"5..\"}[5m])"
done

echo ""
echo "🎉 Complete! $(ls -1 "$OUTPUT_DIR" | wc -l) files in $OUTPUT_DIR/"
echo "📋 Summary: $(ls -lh "$OUTPUT_DIR" | wc -l) files, $(du -sh "$OUTPUT_DIR") total"
ls -lh "$OUTPUT_DIR" | head -10

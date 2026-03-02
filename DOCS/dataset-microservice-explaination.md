# 📊 **SockShop Microservices Anomaly Dataset Documentation**

## **Overview**
High-fidelity **traces + metrics + load data** from SockShop (Weaveworks microservices demo) under **normal vs anomaly conditions**. Collected on GCP VM (12 CPU, 40GB RAM) using **LTTng 2.15 + OpenTelemetry + Prometheus**.

**Goal**: Train ML anomaly detection on distributed tracing data.

## **Dataset Structure**
```
micro-service-trace-data/
├── traces/                    # LTTng CTF traces (~1MB/run)
│   └── normal/run01/
│       ├── kernel/           # Kernel events (syscalls, scheduling)
│       └── ust/              # User-space (OTel spans from relay)
└── experiments/              # Load + metrics (~200KB/run)
    └── normal/run01/
        ├── load_results.csv  # 600-20K requests (latency, errors)
        └── metrics/          # 33 Prometheus JSONs (QPS, P95)
```

## **Trace Types**

### **1. Kernel Traces** (`kernel/kernel/`)
**Low-level system events** during app execution:
```
babeltrace2 kernel/kernel | head
```
```
syscall_entry_epoll_ctl    # Network I/O (epoll)
syscall_entry_read         # Socket reads
net_dev_queue              # Packet TX/RX
timer_hrtimer_cancel       # Timer events
```
**~2M+ events/run** — CPU scheduling, IRQs, block I/O, network stack.

### **2. UST Traces** (`ust/ust/uid/1002/64-bit/`)
**Application spans** from OpenTelemetry → Python relay → LTTng:
```
babeltrace2 ust/ust/uid/1002/64-bit | grep otel.spans | head
```
```
op=find data.cart           # MongoDB cart lookup
op=CartRepository.findByCustomerId  # Spring Boot repo
op=GET /carts/{id}/items    # HTTP endpoint
```
**~4K-20K spans/run** — 40-50% business (`cart`, `orders`), 50% monitoring.

## **How Kernel + UST Relate**
```
Timeline:
1. User → HTTP /carts/123 (UST span starts)
2. Apache → epoll_ctl syscall (Kernel)  
3. Mongo → block I/O (Kernel) → find data.cart (UST)
4. Response → net_dev_queue (Kernel) → span ends (UST)
```
**Kernel shows system bottlenecks** (I/O wait, context switches) causing UST span delays.

## **Experiments Collected**
| Type | Script | Stress | Duration | Requests | Expected |
|------|--------|--------|----------|----------|----------|
| Normal | `normal.sh` | None | 2-5min | 600-20K | 25ms p95 |
| CPU | `cpu_stress.sh` | 12 cores 100% | 3min | 10K+ | 200ms+ |
| Disk | `disk_stress.sh` | 200×50GB writes | 3min | 8K+ | 1s+ timeouts |
| Memory | `mem_stress.sh` | 24GB alloc | 3min | 10K | 300ms GC |

## **Load Generator** (`load_generator.py`)
**Realistic SockShop traffic** (register→cart→orders):
```
POST /register     # User service
GET /catalogue     # Browse socks
POST /cart/items   # Add to cart (carts service)
POST /orders       # Checkout (payment/shipping)
```
**23 scenarios**, weighted for eCommerce patterns.

## **Metrics** (`experiments/*/metrics/*.json`)
**Prometheus queries** (QPS, P95/P50 latency) for:
```
catalogue, orders, payment, shipping, user, frontend
VM: cpu, mem, disk, network
```
**~33 files/run**, 30s step resolution.

## **Analysis Tools**

```bash
# View traces
babeltrace2 traces/normal/run01/kernel/kernel      # Kernel
babeltrace2 traces/normal/run01/ust/ust/uid/*/64-bit  # App spans

# Stats
babeltrace2 ust/ust/uid/*/64-bit | grep otel.spans | grep -c cart
tail -n +2 experiments/normal/run01/load_results.csv | wc -l

# Business purity
babeltrace2 ust/... | grep otel.spans | awk '{print $17}' | grep -i cart | sort | uniq -c
```

## **Scale**
```
~10 normal + anomaly runs
~50MB traces + 2MB experiments
~20K requests/run → 200K total
~50K business spans total
```

**Ready for ML training** — **correlated traces + metrics + load** under realistic stress! 🚀
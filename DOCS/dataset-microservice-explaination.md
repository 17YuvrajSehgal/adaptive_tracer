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

## Apache-comparable evaluation (`train_sockshop.py`)

SockShop OOD metrics are aligned with the Apache LMAT protocol in `functions.ood_detection_ngram`:

- **AUROC** and **AUPR** are computed on a **balanced** test set: `min(n_normal, n_ood)` sequences from `test_id` and `test_ood_{cpu|disk|mem|net}` (same idea as matching normal vs OOD counts on Apache).
- **F1**, **precision**, **recall**, and **accuracy** use a threshold chosen on **validation**: `valid_id` + `valid_ood_{type}` with the same balancing rule; the threshold maximizes F1 over a grid (`--ood_threshold_grid`, default 100). If `valid_id` or `valid_ood_*` is missing, those classification metrics are skipped (`f1_note` in `ood_results.json`) but AUROC/AUPR on the balanced test set are still reported.
- **`--ood_score`**: use `event` or `latency` for single-task runs (must match `--train_event_model` / `--train_latency_model`). Use `combined` for multi-task; mixing uses `--lat_score_weight` when both heads are trained.

**Table 5 (varying duration categories, e.g. 3 / 5 / 7 / 9 bins):** run preprocessing **separately** for each `--n_categories` value so `lat_cat` in the NPZ matches the model (`train_sockshop.py --n_categories` must equal the preprocess setting). Train and evaluate each preprocessed tree; there is no single flag that emits all columns at once.

**Tables 6–7 (Event vs Duration vs Multi-task):** run **three trainings** on the same preprocessed data with different flags:

| Mode | Flags |
|------|--------|
| Event-only | `--train_event_model --ood_score event` |
| Duration-only | `--train_latency_model --ood_score latency` |
| Multi-task | `--train_event_model --train_latency_model --ood_score combined` |

Apache paper rows (Connection, CPU, IO, OPCache, …) map to **stress families**; SockShop rows are **cpu, disk, mem, net** — same reporting structure, different benchmark names.
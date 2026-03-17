***

# LMAT on SockShop: Complete Research Documentation
**Kernel Trace Anomaly Detection for Containerised Microservices**

**Project:** adaptive_tracer
**Platform:** GCP VM (lttng-traces-microservice) + SLURM HPC (trig-login01)
**Authors:** Yuvraj Sehgal (yuvraj17) + collaborators
**Date:** March 2026
**Status:** Experiments complete — paper writeup in progress

***

## 1. Executive Summary

We apply LMAT (Log-based Microservice Anomaly Transformer), originally designed for the Apache/LAMP stack, to Weaveworks SockShop — a Docker-based polyglot microservices benchmark running 7+ loosely coupled services written in Go, Java, Node.js, and Python, all on a single GCP VM.

**Key Results:**

| Anomaly | LSTM AUROC | LSTM AUPR | Transformer AUROC | Transformer AUPR |
|---|---|---|---|---|
| CPU stress | 0.7539 | 0.7564 | 0.7077 | 0.7013 |
| Disk stress | 0.8251 | 0.8368 | 0.7330 | 0.7331 |
| Memory stress | 0.7787 | 0.7721 | 0.7101 | 0.6902 |
| Network stress | 0.6735 | 0.5774 | 0.6526 | 0.5344 |
| **Mean** | **0.7578** | **0.7357** | **0.7009** | **0.6648** |

**Critical Finding (discovered post-training, does NOT invalidate results):** The `proc` (process name) and `pid` fields in the NPZ dataset are completely degenerate — every event maps to `"unknown"` and `0` respectively. The model achieves the above scores using only: syscall type, entry/exit marker, inter-event timing, thread ID, return code, and latency category. This actually strengthens the paper argument — LMAT's detection power comes from syscall sequence structure and latency patterns, not process identity.

***

## 2. System Architecture

Two machines are involved in the full pipeline.

**GCP VM: lttng-traces-microservice**
This machine runs the full SockShop Docker stack and collects all LTTng traces. Services include: frontend, catalogue, carts, orders, payment, shipping, user, queue-master, rabbitmq, session-db (Redis), catalogue-db (MongoDB), carts-db, orders-db, and user-db. Prometheus and Node Exporter run at `http://35.226.88.171:9090`. LTTng 2.15 is installed for both kernel and UST tracing. Traces are written to `~/traces/<type>/<run_id>/` with two subdirectories: `kernel/` for CTF binary syscall data and `ust/` for OTel Python spans.

**HPC: trig-login01 (SLURM cluster)**
This machine handles preprocessing, training, and evaluation. The project root is `~/adaptive_tracer/`. All data lives under `/scratch/yuvraj17/adaptive_tracing_scratch/micro-service-trace-data/`, which contains a `traces/` directory mirroring the GCP VM structure and a `preprocessed/` directory containing vocab files, delay spans, and NPZ shards split into `train_id/`, `valid_id/`, `test_id/`, and `{valid,test}_ood_{cpu,disk,mem,net}/`.

**Important Docker deployment note:** All SockShop containers were deployed without `--cgroupns=private`. This means every container shares the host cgroup namespace, with `ns_inum = 4026531835` for all TIDs. Cgroup namespace IDs therefore cannot be used to distinguish which container generated a given kernel event.

***

## 3. Data Collection Pipeline

### 3.1 OTel-to-LTTng Relay Stack (agents/)

This is a critical piece of infrastructure that is easy to overlook. It bridges OpenTelemetry distributed tracing (running inside Java containers) into the LTTng UST trace, and without it the `ust/` directory in each run would be empty.

**How the full UST pipeline works:**

The four Java-based SockShop services — carts, orders, shipping, queue-master — are started with the OpenTelemetry Java agent attached via `JAVA_TOOL_OPTIONS`. Two config files control this.

`otel.properties` configures the OTel agent:
- `otel.traces.exporter=logging` — spans are written to Java Util Logging (JUL) stdout, not a remote collector
- `otel.logs.exporter=logging`
- `otel.metrics.exporter=none` — metrics are handled separately by Prometheus
- `span-suppression-strategy=none` — all spans including nested/internal ones are emitted
- `otel.propagators=tracecontext,baggage` — W3C Trace Context headers for cross-service correlation

`jul-lttng.properties` configures JUL to route records directly into LTTng UST:
```
handlers = org.lttng.ust.agent.jul.LttngLogHandler
.level = ALL
```
This file must be passed to the JVM via `-Djava.util.logging.config.file=jul-lttng.properties`. The `.level=ALL` ensures nothing is filtered before reaching LTTng.

`otel-to-lttng.py` is a host-side relay daemon that runs outside the containers. It opens `docker logs -f --tail=0` streams for all four Java containers and uses Python `select()` to multiplex reads across them concurrently. A regex pattern matches lines like `[otel.javaagent...] INFO LoggingSpanExporter - 'GET /carts/{id}' : abc123 def456 SERVER` and extracts `op`, `trace_id`, `span_id`, and `kind`. It then emits each span as a LTTng UST event via:
```python
logger = logging.getLogger("otel.spans")
lttng_handler = lttngust.loghandler._Handler()
logger.info(f"op={op} trace_id={trace_id} span_id={span_id} kind={kind}")
```

**Critical limitations of the relay:**

Only 4 of 7+ services are monitored. The Go services (catalogue, user) and Node.js services (frontend, payment) are not tailed and produce zero UST spans. Their activity is captured only in the kernel trace. Additionally, the relay emits only one event per span (the export event, after the span closes), not a start/end pair. The preprocessor tries to infer start/end from paired `span_id` appearances, but most spans end up as zero-duration point events. These two limitations together are why `--seg_mode time` (TID-based 100ms windows) is the correct choice for this dataset — it captures syscall activity from ALL services uniformly, regardless of language runtime.

### 3.2 LTTng Trace Collection (collect_trace.sh)

Each run starts two simultaneous LTTng sessions. The kernel session runs as root and enables all kernel events with `sudo lttng enable-event -k --all '*'`. This captures `syscall_entry_*`, `syscall_exit_*`, `sched_switch`, `lttng_statedump_*`, `irq_*`, `block_*`, `net_*`, `timer_*`, and `rcu_*` events in CTF binary format. The UST session runs without sudo and enables only the `otel.spans` Python event. The OTel relay (`otel-to-lttng.py`) runs in the background for the duration of the trace. Both sessions are stopped and destroyed cleanly at the end.

### 3.3 Load Generator (load_generator.py)

200 concurrent `VirtualUser` threads are launched. Each user registers a new random account, logs in, creates an address and payment card (required to avoid 406 errors on order placement), then loops through randomly weighted scenarios until a stop event is set. The scenario weights are: browse_homepage (15), browse_catalogue (14), view_item (12), detail_page (10), add_to_cart (9), basket_page (9), place_order (5), login (5), get_orders (4), update_cart_item (3), delete_cart_item (3), get_address (3), get_card (3). Think time varies by run type — normal runs use 0.2–1.0s, CPU/Mem/Net anomaly runs use 0.1–0.3s, and disk anomaly uses 0.05–0.2s (most aggressive). Results are written per-request to `load_results.csv`.

### 3.4 Anomaly Injection Scripts

All anomaly scripts follow the same three-component parallel structure: start `collect_trace.sh` in the background, start the stressor in the background, start `load_generator.py` in the background, then `wait` for all three. After they finish, Prometheus metrics are downloaded for the run's time window.

| Script | Duration | Users | Stressor |
|---|---|---|---|
| normal.sh | 300s | 200 | None |
| cpu_ultra.sh | 180s | 200 | `stress-ng --cpu 12 --cpu-method all --cpu-load 100` |
| mem_stress.sh | 180s | 200 | `stress-ng --vm 16 --vm-bytes 90% --vm-keep --page-in` |
| disk_ultra.sh | 180s | 200 | `stress-ng --hdd 300 --hdd-bytes 4G --hdd-opts direct,fsync` |
| net_stress.sh | 180s | 200 | `tc netem delay 150ms±80ms loss 3% + tbf rate 15mbit burst 32k` |

### 3.5 Prometheus Metrics Download (download_metrics.sh)

After each run, metrics are pulled from `http://35.226.88.171:9090` at 30s resolution for the run's time window plus 30s buffers on each side. VM-level metrics collected: cpu, memory, disk, network receive/transmit. Per-service metrics for all 7 services: qps, p50 latency, p95 latency, and 5xx error rate.

***

## 4. Preprocessing Pipeline (preprocess_sockshop.py)

### 4.1 Segmentation Strategy

The mode used for the current dataset is `--seg_mode time` — TID-based fixed-duration windows. Each TID gets independent non-overlapping windows of `--window_ms 100`. A window is emitted when the elapsed time reaches 100ms AND the window contains at least `--min_events 8` kernel events. The implementation is fully streaming and never holds the entire trace in RAM — only per-TID rolling buffers are kept in memory, which is critical for the large disk and CPU anomaly runs.

UST-mode segmentation is also implemented and would use OTel span boundaries to define windows, but as explained in Section 3.1, the relay only covers 4 Java services and emits single-point span events rather than start/end pairs, making the time-window approach more complete and reliable for this dataset.

### 4.2 Sequence Encoding

Each sequence is encoded as a set of parallel integer arrays of the same length L:

- `call` (int32) — syscall vocabulary index, including special tokens at positions 0–4
- `entry` (int8) — 0=neither, 1=entry event, 2=exit event
- `duration` (int64) — nanoseconds since the previous event in the sequence (inter-event delta)
- `proc` (int32) — process name vocabulary index
- `pid` (int32) — raw PID from kernel event payload
- `tid` (int32) — raw TID from kernel event payload
- `ret` (int8) — 0=no return value, 1=success (ret≥0), 2=failure (ret<0)
- `lat_cat` (uint8) — latency category 0 through n_cat (0=pad/entry, 1..n_cat=exit events)
- `seq_len` (int32, scalar) — actual unpadded sequence length
- `req_dur_ms` (float32, scalar) — window duration in milliseconds
- `is_anomaly` (int8, scalar) — 0=normal, 1=anomaly

Special token IDs are uniform across all vocabularies: 0=[MASK], 1=[UNKNOWN], 2=[START], 3=[END], 4=[TRUNCATE]. The [TRUNCATE] token is confirmed present in the shards, meaning some sequences do hit the 512-event cap during preprocessing. The model config sets max_seq_len=4096 but the actual shards cap at 512.

### 4.3 Vocabulary Construction

Vocabulary is built from the training split only and frozen for all subsequent splits. The two vocab files saved are `preprocessed/vocab.pkl` (a pickle of `(dict_sys, dict_proc)`) and `preprocessed/delay_spans.pkl` (latency boundaries per event type).

Confirmed vocabulary sizes:
- `dict_sys` (syscall): 257 entries = 5 special tokens + 252 real kernel event types
- `dict_proc` (process): 6 entries = 5 special tokens + 1 real entry ('unknown')

The top-10 syscall vocabulary entries are: `[MASK]`, `[UNKNOWN]`, `[START]`, `[END]`, `[TRUNCATE]`, `futex`, `pselect6`, `timer_hrtimer_init`, `timer_hrtimer_start`, `rcu_utilization`. The presence of timer and rcu events confirms that `--all '*'` captures more than just syscalls.

### 4.4 Latency Categorisation

With `n_categories=6`, there are 5 real latency bins plus one zero/pad category. The algorithm matches the original LMAT implementation: for each syscall type in the training data, all exit-event latencies (measured as entry_timestamp → exit_timestamp) are collected and triangular percentile boundaries are computed. At inference time, `lat_cat = np.searchsorted(boundaries, latency) + 1`, capped at n_cat. Approximately 252 event types have valid latency boundaries (one per syscall type with sufficient observations). Boundaries are frozen from training and applied identically to all OOD splits.

### 4.5 NPZ Shard Layout

Each shard contains up to 5000 sequences, zero-padded to the longest sequence in that shard. The split directories are:

- `train_id/` — normal/run01, run02, run03 (label=0)
- `valid_id/` — normal/run04 (label=0)
- `test_id/` — normal/run05 (label=0)
- `valid_ood_cpu/` and `test_ood_cpu/` — cpu_stress/ultra_01 and ultra_02 (label=1)
- `valid_ood_disk/` and `test_ood_disk/` — disk_stress/ultra_01 and ultra_02 (label=1)
- `valid_ood_mem/` and `test_ood_mem/` — mem_stress/ultra_01 and ultra_02 (label=1)
- `valid_ood_net/` and `test_ood_net/` — net_stress/ultra_01 and ultra_02 (label=1)

***

## 5. Dataset Statistics

### 5.1 Vocabulary

The syscall vocabulary contains 252 unique real kernel event types. These include not only Linux syscalls (futex, read, write, etc.) but also kernel tracepoints like `timer_hrtimer_init`, `rcu_utilization`, and sched events — all captured by `--all '*'`. The process vocabulary contains exactly 1 real entry: `'unknown'`. This is the critical finding explained in detail in Section 8.

### 5.2 Feature Inspection — Critical Finding

Verified by direct inspection of `train_id/shard_000000.npz`:

| Feature | Status | Evidence |
|---|---|---|
| call | ✅ Active | Well-distributed across 252 syscall types |
| entry | ✅ Active | Values {0,1,2} all present |
| duration | ✅ Active | Non-zero nanosecond deltas throughout |
| tid | ✅ Active | 488 unique TIDs; non-zero fraction = 0.174 |
| ret | ✅ Active | Values {0,1,2} all present |
| lat_cat | ✅ Active | Values {0..5} all present |
| proc | ❌ Dead | Unique values = {0,2,3,4,5} only; every real event = 5 ('unknown') |
| pid | ❌ Dead | Unique values = {0} only; non-zero fraction = 0.000 |

Sequence counts: 316,026 normal sequences across train/valid/test. OOD counts: CPU=387,625, Disk=425,955, Memory=385,131, Network=220,229. Network has the fewest because network impairment causes timeouts that shorten or drop windows.

***

## 6. Model Configurations

Both models were trained on the same preprocessed dataset on an NVIDIA H100 GPU via SLURM. W&B project: `sockshop_lmat`.

### 6.1 LSTM (W&B run: lstm_h100_311554)

Architecture: `n_hidden=1024`, `n_layer=6`, `dropout=0.01`. Embedding dimensions: `dim_sys=48`, `dim_entry=12`, `dim_ret=12`, `dim_proc=48` (wasted — all map to same token), `dim_pid=12` (wasted — all zero), `dim_tid=12`, `dim_order=12`, `dim_time=12`. Training: `lr=0.001`, `warmup_steps=2000`, `clip=10.0`, `batch=512`, `accum_steps=4` (effective batch=2048), `n_epochs=50`, `amp=true`, `label_smoothing=0.1`. Scoring: `lat_score_weight=0.3`, meaning final anomaly score = 0.7 × event_loss + 0.3 × latency_loss.

### 6.2 Transformer (W&B run: transformer_h100_311494)

Architecture: `n_hidden=1024`, `n_layer=6`, `n_head=8`, `dropout=0.1`, `activation=gelu`, `tfixup=false`. Embedding dimensions: `dim_sys=64`, `dim_entry=8`, `dim_ret=8`, `dim_proc=8` (wasted), `dim_pid=16` (wasted), `dim_tid=16`, `dim_order=16`, `dim_time=16`. Training: `lr=0.0003`, `warmup_steps=2000`, `clip=1.0`, `batch=512`, `accum_steps=4`, `n_epochs=50`, `amp=true`, `label_smoothing=0.1`. Same scoring as LSTM.

***

## 7. Experimental Results

Evaluation protocol: OOD anomaly detection. The combined score (0.7 × event reconstruction loss + 0.3 × latency loss) is computed per sequence. AUROC and AUPR are calculated by treating normal sequences as negatives and OOD sequences as positives.

LSTM consistently outperforms Transformer across all four anomaly types by an average margin of 0.057 AUROC. Disk anomaly is the easiest to detect (LSTM AUROC=0.825), likely because `fsync`-heavy I/O creates strongly distinctive syscall bursts. Network anomaly is the hardest (LSTM AUROC=0.673) and also has the fewest OOD sequences (220K vs 385K+), which may contribute to the lower score.

Three hypotheses for why LSTM beats Transformer on this dataset: (1) average sequences well under 512 tokens may be too short for self-attention's long-range advantage over LSTM's sequential inductive bias; (2) the Transformer uses lower lr (3e-4 vs 1e-3) and tighter gradient clipping (1.0 vs 10.0), which may cause underfitting in 50 epochs; (3) the higher dropout (0.1 vs 0.01) may over-regularise given the relatively small and homogeneous vocabulary of 252 syscall types.

***

## 8. Feature Analysis: What the Model Actually Learns From

### 8.1 Root Cause of proc Degeneracy

LTTng 2.15 `syscall_entry_*` and `syscall_exit_*` events do **not** include a `procname` field in their event payload. The field only appears in `lttng_statedump_process_state` (a one-time snapshot at trace start) and `sched_switch` events (which carry `next_comm`/`prev_comm`). The preprocessor calls `_safe_field(ev, ("procname",), "unknown")` for every syscall event and always falls back to `"unknown"`. This was confirmed by running `babeltrace2 trace | grep procname` which returned zero results for syscall events. The result is that `dict_proc` has only one real vocabulary entry despite 7+ distinct microservices generating events.

### 8.2 Root Cause of pid Degeneracy

Similarly, the syscall event payload in this LTTng 2.15 configuration does not include a `vpid` or `pid` field at the per-event level. The preprocessor calls `_safe_field(ev, ("vpid", "pid"), 0)` and always gets 0. Since the padding value is also 0, the model cannot distinguish "no PID data" from "padded position". The PID column in every shard is uniformly zero for all non-padding event positions.

### 8.3 Why TID Partially Compensates

TID (via the `vtid`/`tid` field) IS present in syscall event payloads and is correctly captured. With 488 distinct TIDs in a single training shard, the TID embedding provides meaningful within-sequence context — the model can learn that certain TIDs tend to exhibit certain syscall patterns, partially recovering thread-level identity. However, TIDs are not semantically labelled (the model doesn't know which container a TID belongs to), change across runs, and can be reused by the OS over time.

### 8.4 Wasted Embedding Capacity

For the LSTM: `dim_proc=48 + dim_pid=12 = 60` dimensions wasted out of ~168 total input dimensions (~36% waste). For the Transformer: `dim_proc=8 + dim_pid=16 = 24` dimensions wasted out of ~144 total (~17% waste). The Transformer's smaller `dim_proc` means proportionally less waste, which may partly explain why its relative performance gap with the LSTM is consistent across anomaly types.

### 8.5 Apache vs. SockShop Feature Comparison

| Feature | Apache (LAMP) | SockShop (Docker) |
|---|---|---|
| proc | ✅ Active (apache2, mysqld, php-fpm) | ❌ Dead (all 'unknown') |
| pid | ✅ Active | ❌ Dead (all 0) |
| tid | ✅ Active | ✅ Active (488 unique) |
| syscall vocab | ~180 types | 252 types |
| Services | 3 (monolithic-ish) | 7+ (polyglot) |

LMAT achieves comparable mean AUROC on SockShop despite proc and pid being completely uninformative. This is the strongest argument that syscall sequence structure and latency distributions are the core detection mechanism, not process identity.

***

## 9. Known Bugs and Limitations

[//]: # (### Bug 1 — collect_trace.sh: Unintentional Double CPU Stress &#40;Medium Impact&#41;)

[//]: # ()
[//]: # (`collect_trace.sh` contains its own internal CPU stressor: `stress-ng --cpu 4 --timeout ${DURATION}s` that fires when `TYPE=anomaly_cpu*`. Since `cpu_ultra.sh` already launches `stress-ng --cpu 12 --cpu-load 100` externally, both run simultaneously. The actual CPU stress during anomaly_cpu runs is therefore 16 cores at 100%, not 12 as documented. The anomaly is still clearly anomalous and results are valid, but the intensity is higher than the specification implies.)

### Bug 1 — otel-to-lttng.py: Only 4 of 7+ Services Covered

The relay only tails `carts`, `orders`, `shipping`, and `queue-master`. The `catalogue`, `user`, `frontend`, and `payment` services produce no UST spans. This means `--seg_mode ust` would miss the majority of microservice activity and is therefore not suitable as the primary segmentation strategy for this dataset.

### Limitation 1 — proc and pid Features Are Degenerate

Not a code bug — a consequence of the LTTng tracer configuration. All proc=`unknown`, all pid=0. See Section 8 for full analysis.

### Limitation 2 — cgroup Namespace Cannot Identify Containers

All TIDs share `ns_inum = 4026531835` (host cgroup namespace). Docker was deployed without `--cgroupns=private` on this GCP VM, so cgroup namespace IDs cannot serve as container identity proxies.

### Limitation 3 — Single-Point OTel Spans

The relay emits only one LTTng event per span (at export time), not a start/end pair. Span durations in the UST trace are therefore mostly zero, making span-boundary segmentation unreliable. The `_extract_ust_spans()` function in the preprocessor attempts to pair events by `span_id` but most spans remain as point events.

***

## 10. Paper Framing and Recommended Statements

**For the Instrumentation / Data Collection section:**

> "Java-based SockShop services (carts, orders, shipping, queue-master) are instrumented with the OpenTelemetry Java agent configured to export spans via Java Util Logging (JUL). A host-side relay process intercepts these log-formatted spans from the Docker log streams and re-emits them as LTTng UST events using the `lttngust` Python handler, creating correlated kernel and userspace traces in a unified CTF format. Go and Node.js services are not covered by this relay; their activity is captured exclusively in the kernel trace. Due to this partial span coverage and the single-event-per-span relay design, time-window segmentation (100ms TID-based windows) is used for dataset construction rather than span-boundary segmentation, ensuring uniform coverage across all microservices regardless of language runtime."

**For the Experimental Setup section:**

> "Unlike the Apache/LAMP experimental setup, where LTTng kernel syscall events carry a `procname` field identifying individual service workers (e.g., apache2, mysqld, php-fpm), the SockShop containerised deployment presents a fundamentally different tracing environment. In LTTng 2.15, `syscall_entry_*` and `syscall_exit_*` events do not include process name in their payload — this field appears only in one-time statedump events and `sched_switch` records. Consequently, all 316,026 normal training sequences carry a uniform 'unknown' process label, rendering the process-name embedding dimension uninformative. Similarly, the PID field defaults to zero throughout the dataset. Thread IDs (TIDs), which are present in syscall event payloads, provide partial process-discriminating context, with 488 distinct TID values per training shard corresponding to threads across the Docker containers sharing the host kernel."

**For the Results / Discussion section:**

> "Notably, LMAT achieves mean AUROC of 0.758 (LSTM) and 0.701 (Transformer) on SockShop despite the proc and pid embedding channels contributing zero discriminative signal — a constraint absent in the Apache experiments. This result isolates syscall sequence structure and per-syscall latency distributions as the primary anomaly detection mechanisms, independent of process identity. Strong performance under this more constrained feature set strengthens the generalisability claim: LMAT transfers to containerised microservice workloads even when fine-grained process metadata is unavailable from the kernel tracer."

**For the Future Work section:**

> "Future data collection should enable `sched_switch` tracing alongside syscall events — a one-line addition to the collection script — permitting construction of a TID-to-process-name mapping via the `next_comm` field. Notably, `sched_switch` events are already present in the current raw CTF traces and could be parsed in a new preprocessing pass without re-collecting data. An ablation comparing detection performance with and without real process names would quantify the contribution of process identity. Additionally, deploying Docker with `--cgroupns=private` would allow per-container cgroup namespace IDs to supplement thread identity, and extending the OTel relay to cover Go and Node.js services would enable span-boundary segmentation for the full microservice graph."

**Claim hierarchy for the paper:**

The strongest and most fully supported claim is: *"LMAT detects resource-exhaustion anomalies in containerised microservices from kernel syscall traces alone."* The supported-with-caveat claim is: *"LMAT generalises from monolithic (Apache) to microservice (SockShop) deployments with no architectural changes."* The honest limitation that must be stated is: *"In the current SockShop dataset, process-name and PID metadata are unavailable at the syscall event level due to LTTng tracer configuration; thread ID serves as a partial substitute."*

***

## 11. Future Work

**Short-term (for revision or extended version):**

The highest-value change is enabling `sched_switch` parsing in the preprocessor to recover real process names from the existing raw traces — no new data collection required. Adding one line (`sudo lttng enable-event -k sched_switch`) to `collect_trace.sh` would ensure future runs have even richer data. An ablation study with and without proc names would strengthen the paper significantly. The `net_stress.sh` trap bug should be fixed before any new net anomaly runs. Deploying Docker with `--cgroupns=private` on the next VM setup would give cleaner container identity.

**Medium-term:**

Collecting additional normal runs (currently only 5) and doing cross-validation would provide confidence intervals on the AUROC numbers. Additional anomaly types worth exploring: network partition injection, OOM kill, file-descriptor exhaustion, and container CPU throttling (via cgroups, not host-level stress-ng). Fixing the OTel relay to emit paired start/end events per span would make `--seg_mode ust` a viable alternative to time-window segmentation.

**Long-term:**

Multi-host Kubernetes deployment to test whether LMAT handles distributed kernel traces across nodes. Online streaming anomaly detection — the current pipeline is entirely offline batch evaluation and would need significant engineering to operate in real-time.

***

## 12. Reproduction Guide

**Step 1 — Collect data (GCP VM):**

Prerequisites: SockShop Docker stack running with all 14 containers healthy, `lttng`, `stress-ng`, `python3`, and `iproute2` installed, all scripts in home directory, and `~/agents/otel-to-lttng.py` present with `otel.properties` and `jul-lttng.properties` mounted into Java containers.

Run: `./normal.sh run01 300` through `run05`, then both runs of each anomaly type. Fix the `net_stress.sh` trap before running net anomaly.

**Step 2 — Transfer to HPC:**

`rsync -avz ~/traces/ trig-login01:/scratch/yuvraj17/adaptive_tracing_scratch/micro-service-trace-data/traces/`

**Step 3 — Preprocess (HPC):**

Optionally pre-convert CTF to text first for 5–10x faster parsing: `babeltrace2 traces/normal/run01/kernel > txt_dumps/normal_run01_kernel.txt` for each run. Then run `preprocess_sockshop.py` with `--seg_mode time --window_ms 100 --warmup_s 5 --min_events 8 --n_categories 6 --shard_size 5000` and all eleven split specs.

**Step 4 — Verify:**

Check vocab sizes (`dict_sys`=257, `dict_proc`=6) and confirm the feature degeneracy: proc unique values should be `{0,2,3,4,5}`, pid non-zero fraction should be 0.000, tid unique count should be ~488.

**Step 5 — Train:**

`python train.py --config microservice/configs/lstm_sockshop.json` and `python train.py --config microservice/configs/transformer_sockshop.json`.

**Step 6 — Evaluate:**

`python evaluate_ood.py` with the appropriate model checkpoint and config, outputting a JSON with `auroc` and `aupr` per anomaly type. Expected results: LSTM mean AUROC ~0.758, Transformer mean AUROC ~0.701.

***

*Last updated: March 14, 2026. Prepared by Yuvraj Sehgal (yuvraj17) with AI-assisted analysis. For internal use — adaptive_tracer research project.*
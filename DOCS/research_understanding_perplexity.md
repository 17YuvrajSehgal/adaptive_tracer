# LMAT Adaptive Tracing Repository and SockShop Microservice Experiments – Technical Understanding

## Executive Summary

The adaptive_tracer repository implements LMAT, a language-model-based adaptive tracing framework that models kernel event streams as a sequence language to enable anomaly detection and adaptive observability. 
Recent extensions in the DOCS directory document a full microservice evaluation on the SockShop benchmark, directly addressing reviewer concerns about generalizability beyond the original Apache/LAMP setting. 
The SockShop experiments show that LMAT’s LSTM and Transformer variants detect CPU, disk, memory, and network stress anomalies from kernel traces alone with mean AUROC around 0.75 on a containerised microservice deployment, strengthening the claim that detection power comes primarily from syscall sequence and latency structure. 
The attached reviewer comments emphasize two main revision axes: (1) validation on architecturally distinct systems such as microservices, and (2) explicit quantification of LMAT’s overhead on the monitored system, both of which are now being actively addressed.[^1]

## Core LMAT Concept and Goals

LMAT (Language Model–based Adaptive Tracing) treats kernel-level events (primarily syscalls and related tracepoints) as a discrete language and applies sequence models (LSTM and Transformer) for next-event and duration prediction. 
By training models on normal behavior, LMAT uses reconstruction/forecasting loss as an anomaly score, which drives an adaptive tracing policy that keeps overhead low in normal periods and escalates trace granularity upon detected deviations. 
The framework further exposes root-cause signals via event-level error vectors and top-attribution events, along with an operator feedback loop that clusters benign novelties so that users can label entire groups once, reducing false positives over time. 
A key design constraint is practical footprint: the production-targeted LMAT variants are around one million parameters so they can run on CPU or modest GPUs instead of datacenter-scale accelerators, aligning with realistic DevOps deployments. 

## Original Apache/LAMP Setting (High-Level)

The initial LMAT evaluation (summarised in the understanding doc rather than full Apache docs) targets an Apache web server (LAMP stack), where traces are segmented at request level and include active process and PID metadata at syscall granularity. 
In that environment, proc and pid features are informative—distinguishing apache2 workers, database daemons, and PHP-FPM processes—so the model can leverage both control-flow patterns and process identities for anomaly detection. 
Reviewers noted that while the single-system LAMP evaluation is convincing for monolithic-style workloads, it does not by itself validate claims about broader distributed or microservice environments, motivating the new SockShop experiments.[^1]

## Repository Structure and Key Components

The repository root contains the project README, core training and evaluation code, model implementations, dataset utilities, SLURM scripts, and a large trace dataset archive; the DOCS directory holds higher-level methodological and experimental documentation. 
Core logic resides in `main.py` and `functions.py`, which implement data ingestion from trace dumps, segmentation into sequences, multi-task training loops, adaptive tracing evaluation routines, and an n-gram baseline. 
The `models/` directory contains the multi-feature embedding layer, LSTM and Transformer architectures, custom multi-head attention, SwiGLU activation, and label-smoothing loss definitions, all wired for joint event and duration prediction. 
The `dataset/` package provides dictionary/vocabulary management and an IterableDataset that streams NPZ shards from disk, supports variable-length sequences, truncation, and distributed training via PyTorch’s DataLoader. 
SLURM scripts in `scripts/` encode different hyperparameter settings for LSTM and Transformer runs, including learning rates, dropout, gradient clipping, and maximum updates, targeting both single- and multi-GPU training on an HPC cluster. 

## Event Representation and Feature Set

Each preprocessed sequence is represented as parallel arrays encoding syscall identity, entry/exit markers, inter-event timing, process/thread identifiers, return codes, latency category, and scalar metadata such as sequence length and request/window duration. 
Special tokens `[MASK]`, `[UNKNOWN]`, `[START]`, `[END]`, and `[TRUNCATE]` occupy indices 0–4 in every vocabulary, with real syscall and process tokens following, and a TRUNCATE token explicitly marking sequences that hit the per-shard length cap during preprocessing. 
In the SockShop microservice dataset, the syscall vocabulary contains 252 distinct kernel event types beyond the five special tokens, including both classic syscalls (e.g., `read`, `write`, `futex`) and tracepoints such as timer and RCU events due to enabling all kernel events. 
A critical data finding is that `proc` and `pid` channels are effectively dead features in the microservice dataset: all real events map to a single `'unknown'` process token and all PIDs are zero, leaving TID, syscall identity, and latency categories as the main discriminative signals. 

## Multi-Task Modeling and Training Strategy

LMAT uses multi-task learning with two heads: one predicts the next syscall token and the other predicts a duration/latency category, combining their cross-entropy losses with configurable weights. 
Typical training uses Adam with label smoothing, gradient clipping, mixed precision (AMP), and large effective batch sizes achieved by gradient accumulation, allowing long sequences (up to 4096 tokens) to be processed efficiently on a single H100 GPU. 
The LSTM configuration for SockShop uses 6 layers with hidden size 1024, low dropout (0.01), and multi-feature embeddings that devote substantial dimensions to syscall ID, entry/exit, return code, TID, time encoding, and (nominally) process and PID, although those last two are wasted in this dataset. 
The Transformer variant also employs 6 layers with 8 attention heads, hidden size 1024, dropout 0.1, and smaller embedding dimensions for the degenerate proc/pid features, partly mitigating their capacity waste. 
An `ordinal_latency` option adds a continuous ordinal penalty on latency buckets, shifting the loss landscape and forcing the models to focus on relative distance between categories rather than exact bucket matches, which particularly benefits the Transformer in later experiments. 

## Anomaly Scoring and OOD Evaluation Protocol

At evaluation time, the model processes entire sequences and, at each position, produces distributions over the next syscall and current duration category; per-token cross-entropy against the observed events yields token-level surprise scores. 
These token-level event and latency losses are averaged over valid tokens in each sequence to form mean event CE and mean latency CE, which are then combined as a weighted sum, for example score = 0.7 × event_CE + 0.3 × latency_CE. 
This per-sequence score serves as an unsupervised anomaly score: high values indicate that the observed syscall-language and timing are unlikely under the model’s learned normal distribution, and low values indicate typical behavior. 
Anomaly detection performance is quantified by AUROC and AUPR over large sets of normal vs. out-of-distribution (OOD) sequences, without choosing a fixed threshold, giving a threshold-free view of discriminative power under class imbalance. 

## Reviewer Feedback and Motivation for Microservice Experiments

The editor’s meta-review and reviewer reports emphasise two shortcomings in the earlier Apache-centric submission: limited architectural generalizability and incomplete overhead analysis on the monitored system.[^1]
Reviewer 3 in particular argued that merely adding another physical machine running the same LAMP stack demonstrates hardware portability, not diversity of workloads; they requested evaluation on stream-processing engines or microservice benchmarks such as TrainTicket or Google Hipster Shop.[^1]
The same reviewer also criticised the overhead analysis for reporting only model inference latency (e.g., ~19 ms) rather than end-to-end impact on application latency and throughput, calling for P95/P99 response time and throughput comparisons with and without LMAT enabled.[^1]
Reviewers 1 and 2 were more positive overall but still called for either explicit scoping of claims to single-host systems or additional experiments that demonstrate LMAT’s usefulness in distributed or microservice environments.[^1]
These comments directly motivate the new SockShop microservice experiments documented in DOCS, which aim to validate LMAT on an architecturally distinct, containerised workload while also collecting metrics suitable for overhead analysis. 

## SockShop Microservice Platform and Instrumentation

SockShop is a polyglot microservice benchmark consisting of more than seven loosely coupled services (frontend, catalogue, carts, orders, payment, shipping, user, and multiple backing databases and queues) deployed as Docker containers on a single GCP VM. 
The GCP VM is provisioned with 12 CPU cores, 40 GB RAM, and 100 GB SSD, running Ubuntu 24.04 with Docker 27.0 and LTTng 2.15 as the tracing stack. 
Java-based SockShop services (carts, orders, shipping, queue-master) are instrumented with the OpenTelemetry Java agent, configured to export traces via Java Util Logging (JUL) rather than a remote collector. 
A host-side Python relay (`otel-to-lttng.py`) tails Docker logs for these Java services, parses OTel-formatted log lines to extract operation names and trace/span IDs, and re-emits them as LTTng UST events via the `lttngust` Python log handler. 
LTTng sessions are created per run: a user-space session capturing `otel.spans` and a kernel session capturing all kernel events (`sudo lttng enable-event -k --all '*'`), with outputs stored in separate `ust/` and `kernel/` subdirectories under each run. 

## Microservice Trace and Dataset Layout

Raw data is organised under a `micro-service-trace-data` root (on the HPC scratch filesystem), with a `traces/` tree mirroring the GCP VM’s trace directories and an `experiments/` tree containing load and Prometheus metric files per run. 
For each experiment (normal or stressed), `traces/<type>/<run_id>/kernel` holds the CTF binary kernel events and `traces/<type>/<run_id>/ust` holds the UST OTel span events created by the relay. 
The `experiments/<type>/<run_id>/load_results.csv` file records each synthetic user request with timestamp, scenario, HTTP method, endpoint, status code, latency, and success flag, while `experiments/<type>/<run_id>/metrics/` stores about 33 Prometheus JSON files with QPS and latency metrics per service and host-level resource metrics. 
Prometheus metrics are scraped at 30-second resolution over each run’s time window with additional buffers, providing VM-level CPU, memory, disk, and network data as well as per-service p50/p95 latency and error rates for the main microservices. 
Overall dataset scale is on the order of 10+ runs spanning normal and anomaly conditions, roughly 50 MB of traces plus about 2 MB of experiment metadata, with around 200K HTTP requests and 50K business-relevant spans in total. 

## Load Generation and Anomaly Injection

A custom `load_generator.py` script launches hundreds of virtual users (e.g., 200 threads) that perform realistic SockShop workflows, including registration, login, browsing the catalogue, adding items to the cart, and placing orders. 
Scenarios are weighted to mimic typical e-commerce behaviour, with higher probabilities for browsing and cart interactions than for checkout, and think times vary by run type, with anomaly runs often using shorter think times to increase load intensity. 
Anomaly injection scripts follow a standard pattern: they spawn LTTng trace collection, invoke a `stress-ng`-based stressor (CPU, memory, disk) or traffic control–based network impairment, and run the load generator concurrently, then stop all components and fetch Prometheus metrics. 
CPU stress uses commands like `stress-ng --cpu 12 --cpu-method all --cpu-load 100`, memory stress allocates large virtual memory regions, disk stress issues many concurrent write operations with options like `direct,fsync`, and network stress applies delay and loss using `tc netem` combined with token bucket filters. 
Normal runs omit the stressors and instead exercise the system under a more moderate, but still substantial, request rate (on the order of 1000+ requests per second aggregated across users). 

## Preprocessing and Segmentation for SockShop

For SockShop, the chosen segmentation mode is time-based TID windows: events are grouped per thread ID into fixed 100 ms non-overlapping windows, emitting a sequence when the time budget is reached and the window contains at least a minimum number of events (e.g., 8). 
This streaming segmentation algorithm maintains per-TID rolling buffers without loading full traces into memory, which is crucial given the millions of kernel events per run, especially under disk and CPU stress where event rates spike. 
UST-based segmentation by span boundaries is implemented in the code but is not used for this dataset because the relay only covers four Java services and emits single-point span events rather than proper start–end pairs, making time windows a more complete and robust representation of multi-language microservice activity. 
After segmentation, sequences are encoded into NPZ shards of up to 5000 sequences each, zero-padded to the shard’s longest sequence, and split into train/validation/test ID sets plus OOD splits for each anomaly type (CPU, disk, memory, network) with separate valid and test subsets. 
The shard directories like `train_id`, `valid_id`, `test_id`, `valid_ood_cpu`, and `test_ood_cpu` together span over 300K normal sequences and several hundred thousand OOD sequences per anomaly type, providing ample data for robust evaluation. 

## Feature Degeneracy and Its Implications

Direct inspection of SockShop shards reveals that syscall identity (`call`), entry marker, inter-event duration (`duration`), TID, return code, and latency category are all active features with healthy distributions and nontrivial variance. 
In contrast, the `proc` feature has only one real value (`'unknown'`), and `pid` is uniformly zero for all non-padding positions, a consequence of the specific LTTng 2.15 configuration where syscall events lack per-event process name or PID fields. 
These degenerate channels mean that a significant portion of the embedding dimensions devoted to proc and pid (especially in the LSTM configuration) carry no discriminative information, effectively wasting capacity. 
Thread ID partially compensates for the missing per-event process identity: there are hundreds of unique TIDs in a single shard, and the model can learn that certain TIDs correspond to characteristic syscall patterns, even though TIDs are anonymous and not stable across runs. 
Comparing with the original Apache experiments, where proc and pid were active and could distinguish web server workers and database processes, SockShop presents a strictly more challenging feature setting, making LMAT’s good performance here a stronger demonstration of its reliance on sequence and timing structure. 

## Model Configurations and Training Outcomes

On SockShop, both LSTM and Transformer models are trained on the same NPZ dataset using H100 GPUs with bf16 mixed precision and an effective batch size of 2048 sequences (batch 512 with 4-step accumulation) and sequence length cap 4096. 
In the 50-epoch joint-loss runs (next-event + latency bucket), the LSTM attains lower training and validation loss than the Transformer and notably higher latency bucket accuracy (around 99.6% vs. mid-70s), reflecting better capture of fine-grained timing patterns. 
For OOD detection at 50 epochs, the LSTM achieves mean AUROC ≈ 0.758 and mean AUPR ≈ 0.736 across CPU, disk, memory, and network anomalies, while the Transformer lags at mean AUROC ≈ 0.701 and mean AUPR ≈ 0.665. 
Extending training to 100 epochs with `--ordinal_latency` substantially boosts Transformer performance (mean AUROC ≈ 0.736, mean AUPR ≈ 0.716) and modestly adjusts LSTM results (AUROC ≈ 0.754, AUPR ≈ 0.743), leaving LSTM still ahead overall, especially in high-precision operating regimes. 
Training time roughly doubles when going from 50 to 100 epochs; the Transformer becomes competitive only in the longer, more expensive setting, whereas the LSTM delivers strong performance already at 50 epochs with lower training time. 

## Anomaly-Type-Specific Behaviour

Disk anomalies are consistently the easiest to detect: in the base 50-epoch setting the LSTM achieves AUROC ≈ 0.825 and AUPR ≈ 0.837, reflecting the highly distinctive I/O patterns and latency spikes induced by aggressive disk stress. 
CPU and memory anomalies show intermediate difficulty, with AUROC values in the mid-0.7 range and strong AUPR, indicating that changes in scheduling and memory behaviour produce noticeable shifts in syscall interleavings and durations. 
Network anomalies are the hardest, with AUROC around 0.673–0.668 and substantially lower AUPR; many aspects of network impairment are handled by the kernel’s internal networking stack without dramatic changes in user-visible syscall sequences, making anomalies subtler in this feature space. 
The dataset also contains fewer OOD sequences for network stress than for CPU/disk/memory, which may further contribute to slightly weaker anomaly separability due to limited coverage of atypical patterns. 

## Interpretation of Microservice Results for Generalizability

The SockShop experiments demonstrate that LMAT can transfer from a monolithic Apache/LAMP setup to a containerised multi-service workload without architectural changes to the model, still achieving strong OOD detection performance across multiple resource anomaly types. 
Because proc and pid are unusable in the current SockShop traces, the results isolate kernel syscall sequence and latency structure as core detection mechanisms, independent of explicit process identity, strengthening the argument that LMAT operates on application-agnostic kernel behaviour. 
However, the current microservice setup still runs on a single physical host, so while it addresses architectural diversity at the software level (polyglot microservices vs. monolithic web server), it does not yet realise fully distributed multi-host tracing scenarios that reviewers also hinted at.[^1]
The honest claim hierarchy reflected in the documentation is that LMAT robustly detects resource-exhaustion anomalies in containerised microservices from kernel syscall traces alone and generalises from monolithic to microservice deployments on a single host, with multi-host distributed tracing left for future work. 

## Known Limitations and Potential Fixes

The documentation explicitly notes that only four of the seven-plus SockShop services are covered by the OTel-to-LTTng relay; Go and Node.js services produce no UST spans, limiting the usefulness of span-based segmentation for a holistic distributed view. 
The absence of process name and PID in syscall events is attributed to LTTng 2.15’s event payload definitions; procname appears only in statedump and `sched_switch` events, suggesting that a preprocessing pass which correlates `sched_switch` with syscalls could recover real process names without recollecting traces. 
cgroup namespaces are not usable for container identity in the current GCP VM because Docker was run without `--cgroupns=private`, causing all containers to share the host cgroup namespace ID, so container-level attribution from kernel events alone is weak. 
The relay emits a single event per OTel span at export time, rather than start/end pairs, so span durations in UST are mostly zero; extending the relay to emit explicit start and end events would enable more precise span-boundary segmentation in future datasets. 
The documentation suggests adding `sched_switch` tracing, enabling Docker private cgroup namespaces, expanding OTel coverage to Go/Node services, and collecting additional normal and anomaly runs as concrete steps to strengthen both feature richness and statistical robustness. 

## Reproduction and Engineering Pipeline

The end-to-end pipeline starts with data collection on the GCP VM via shell scripts (`normal.sh`, `cpu_ultra.sh`, `mem_stress.sh`, `disk_ultra.sh`, `net_stress.sh`) that orchestrate LTTng, stressors, and load generation, then syncs traces to an HPC cluster using rsync. 
On the HPC cluster, optional CTF-to-text conversion with babeltrace2 accelerates preprocessing, after which `preprocess_sockshop.py` is run with specific segmentation and latency-category parameters to generate NPZ shards and vocab/delay-span pickles. 
Training is performed via SLURM scripts pointing to JSON configs for LSTM and Transformer models, and evaluation scripts compute AUROC/AUPR per anomaly type using the best validation-loss checkpoint. 
The documentation enumerates concrete expected values—vocab sizes, feature degeneracy statistics, and approximate AUROC/AUPR ranges—that can be used as sanity checks when reproducing the experiments in a fresh environment. 

## Alignment with Reviewer Requests and Next Directions

By moving from an Apache-only workload to SockShop, the project now includes an evaluation on an architecturally distinct, containerised microservice benchmark, directly addressing Reviewer 3’s call for diversity beyond a standard request–response web server.[^1]
The dataset design—combining kernel traces, partial application spans, load logs, and Prometheus metrics—also lays the groundwork for studying LMAT’s impact on p95/p99 latency and throughput by comparing metrics across runs with and without LMAT enabled, which is essential for a complete overhead analysis. 
Documentation explicitly recommends further work on sched_switch-based process reconstruction, cgroup-based container identity, extended OTel coverage, and multi-host Kubernetes deployments to push LMAT toward truly distributed tracing scenarios. 
These directions map cleanly onto the reviewers’ major concerns (generalizability and overhead), positioning the current microservice experiments as a strong intermediate step and a base for additional experiments the user now plans to design.[^1]

---


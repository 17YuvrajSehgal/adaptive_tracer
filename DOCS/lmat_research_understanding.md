# LMAT Research Understanding

## Overview

**LMAT (Language Model-based Adaptive Tracing)** is a research framework for adaptive, low-overhead system observability that uses language models to analyze kernel-level event streams.

### Core Concept
The system treats kernel events (system calls) as a "language" and applies sequence modeling techniques (LSTM, Transformer) to:
1. **Predict next events** (next-event prediction task)
2. **Predict event durations** (duration/latency prediction task)
3. **Detect behavioral anomalies** (out-of-distribution detection)
4. **Provide root-cause analysis** (error vectors and attribution)

### Research Status
- **Paper**: "LMAT: An Adaptive Tracing Approach Based on Efficient System Behavior Analysis Using Language Models"
- **Journal**: The Journal of Systems & Software
- **Status**: Under revision (Major Revision requested - Feb 2026)
- **Manuscript ID**: JSSOFTWARE-D-25-01126R1

---

## Key Research Contributions

### 1. Adaptive Tracing
- Dynamically raise/lower trace detail based on live deviation detection
- Low overhead during normal operation
- Increased granularity when anomalies detected

### 2. Multi-Task Modeling
- **Task 1**: Next-event prediction (which system call comes next)
- **Task 2**: Duration prediction (how long will the event take)
- Joint training improves both tasks

### 3. Root-Cause Analysis
- Error vectors highlighting anomalous events
- Top-attribution events for debugging
- Helps operators understand *why* something is anomalous

### 4. Operator Feedback Loop
- Label-once mechanism via clustering of benign novelties
- Reduces false positives over time
- Learns from operator feedback

### 5. Practical Footprint
- 1M-parameter models (LSTM and BERT-style Transformer)
- Runs on CPU or modest GPU
- Designed for production deployment

---

## Architecture & Components

### Data Pipeline

#### 1. Trace Collection
- Uses **LTTng** (Linux Trace Toolkit Next Generation) for kernel tracing
- Captures system call events with metadata:
  - System call name (e.g., `read`, `write`, `poll`)
  - Entry/exit markers
  - Process name, PID, TID
  - Return values (success/failure)
  - Timestamps and durations

#### 2. Request-Based Segmentation
The system segments traces into **requests** (e.g., HTTP requests for Apache web server):
- Each request is a sequence of kernel events
- Bounded by special tokens: `[START]` and `[END]`
- Enables request-level analysis

#### 3. Event Representation
Each event is represented by multiple features:
```
Event = {
    call:      system call ID (vocabulary index)
    entry:     0=none, 1=entry, 2=exit
    duration:  time since previous event
    proc:      process name ID
    pid:       process ID
    tid:       thread ID
    ret:       0=no return, 1=success, 2=failure
    latency:   event-specific latency
    f_mean:    feature mean (statistical)
}
```

#### 4. Duration Categorization
Latencies are categorized into bins (4, 6, or 8 categories):
- Computed using percentile-based boundaries
- Separate boundaries per event type
- Enables ordinal classification

#### 5. Data Format
- Stored as text files (`data.txt`)
- Format: semicolon-separated sequences
- Example: `call_seq;entry_seq;duration_seq;proc_seq;...;timestamp;duration_tags;request_duration`

### Model Architectures

#### LSTM Model
```python
class LSTM(nn.Module):
    - Embedding layer (multi-feature)
    - LSTM layers (2 layers, 256 hidden units)
    - Two output heads:
      1. Event classifier (next system call)
      2. Latency classifier (duration category)
```

**Key Features**:
- Input: Multi-feature embeddings (call, entry, ret, time, proc, pid, tid)
- Hidden size: 256
- Layers: 2
- Dropout: 0.01
- Total params: ~1M

#### Transformer Model
```python
class Transformer(nn.Module):
    - Embedding layer (multi-feature)
    - Transformer encoder layers
    - Two output heads:
      1. Event classifier
      2. Latency classifier
```

**Key Features**:
- Multi-head attention
- T-Fixup initialization (optional)
- Causal masking (autoregressive)
- Same dual-task output as LSTM

### Training Strategy

#### Multi-Task Learning
```python
# Joint loss function
total_loss = α * event_loss + β * latency_loss

# Event loss: Cross-entropy for next-event prediction
event_loss = CrossEntropyLoss(predictions, next_events)

# Latency loss: Cross-entropy for duration categories
# (or MSE for continuous, BCE for ordinal)
latency_loss = CrossEntropyLoss(duration_preds, duration_categories)
```

#### Training Configuration
From [`scripts/lstm-1.sh`](file:///c:/workplace/research/adaptive_tracer/scripts/lstm-1.sh):
- Optimizer: Adam
- Learning rate: 0.001
- Label smoothing: 0.1
- Batch size: 16
- Gradient clipping: 10
- Mixed precision (AMP): enabled
- Early stopping patience: 20 epochs
- LR reduction patience: 5 epochs
- Max updates: 1,000,000

---

## Evaluation Methodology

### Dataset Structure

#### In-Distribution (ID)
- **Train ID**: Normal Apache web server behavior
- **Valid ID**: Validation set (same distribution)
- **Test ID**: Test set (same distribution)

#### Out-of-Distribution (OOD)
Multiple anomaly types:
1. **Connection anomalies**: Network issues
2. **CPU anomalies**: High CPU load
3. **IO anomalies**: Disk I/O problems
4. **OPCache anomalies**: PHP opcode cache issues
5. **Socket anomalies**: Socket-related problems
6. **SSL anomalies**: SSL/TLS issues (test only)

### Adaptive Tracing Evaluation

The [`adaptive_tracing_eval`](file:///c:/workplace/research/adaptive_tracer/functions.py#L2239-L3098) function performs:

#### 1. Anomaly Detection
- Compute per-request loss on validation/test sets
- Compare ID vs OOD distributions
- Use loss as anomaly score

#### 2. Threshold Selection
- Determine threshold on validation OOD
- Apply to corresponding test OOD
- Evaluate AUROC, F1, precision, recall

#### 3. Root-Cause Analysis
- Generate error vectors (per-event loss contributions)
- Identify top-K most anomalous events
- Provide interpretable debugging signals

#### 4. Metrics
- **Event prediction**: Accuracy, perplexity
- **Duration prediction**: Accuracy, MAE
- **OOD detection**: AUROC, F1, precision, recall
- **Root-cause**: Top-K attribution accuracy

---

## Key Implementation Details

### Data Loading
[`IterableDataset`](file:///c:/workplace/research/adaptive_tracer/dataset/IterableDataset.py):
- Streams data from disk (no full load into memory)
- Supports distributed training (DDP)
- Handles variable-length sequences
- Truncates to `max_token` if needed

### Distributed Training
Uses PyTorch `DistributedDataParallel`:
```python
mp.spawn(train, nprocs=len(gpus), args=(...))
```
- Multi-GPU support
- Gradient synchronization
- Efficient batch processing

### Loss Functions
- **Event**: CrossEntropyLoss (ignore padding)
- **Latency**: 
  - Categorical: CrossEntropyLoss
  - Ordinal: BCEWithLogitsLoss
  - Continuous: MSELoss

---

## Reviewer Feedback & Open Issues

### Critical Concerns (Revision Required)

#### 1. Generalizability (Major Issue)
**Problem**: Only evaluated on Apache web server (LAMP stack)
- Reviewers want validation on different workload types
- Suggested: Redis, Kafka, microservices (TrainTicket, Hipster Shop)
- Current: Only tested on 2 physical machines with same LAMP stack

**Action Required**: Evaluate on architecturally distinct systems

#### 2. Performance Overhead (Major Issue)
**Problem**: Missing impact measurement on target system
- Current: Only reports model inference latency (~19ms)
- Missing: P95/P99 latency impact on Apache server
- Concern: Model can't keep up at 1,000 req/s

**Action Required**: Measure application performance degradation (response time, throughput)

#### 3. Distributed Systems Limitation
**Problem**: Unclear applicability to distributed/multi-tenant systems
- Current: Single-host kernel traces
- Distributed tracing is much more complex (event ordering across machines)
- Abstract/intro may oversell applicability to "DevOps"

**Action Required**: Either validate on distributed systems OR explicitly scope to monolithic/single-host systems

### Minor Issues
- Cost-benefit analysis needs revision after overhead measurement
- Related work section improved (added Sieve, TraStrainer)
- Anomaly generation methodology clarified

---

## Technical Highlights

### 1. Multi-Category Duration Modeling
The system supports multiple granularities (4, 6, 8 categories):
```python
# From data generation
cat_idx = 0 if n_categories == 4 else 1 if n_categories == 6 else 2 if n_categories == 8 else 3
```
This allows trading off between precision and model complexity.

### 2. Ordinal Latency Encoding
For ordinal classification:
```python
# Transform: category 3 → [1,1,1,0,0,0] (for 8 categories)
y_ordinal = transform_ordinal(y_latency, n_category)
```
Preserves ordering information in the loss.

### 3. Continuous Latency Normalization
```python
# Per-event-type min-max normalization
y_normalized = (y - min_vals[event_id]) / (max_vals[event_id] - min_vals[event_id])
```
Handles wide range of latency scales.

### 4. Request-Level Aggregation
```python
# Average per-event loss over request
request_loss = sum(event_losses) / num_events
```
Provides request-level anomaly scores.

---

## File Structure

```
adaptive_tracer/
├── README.md                    # Project overview
├── reviewer_comments.txt        # Journal revision feedback
├── main.py                      # Entry point
├── functions.py                 # Core logic (142KB, 3680 lines)
├── requirements.txt             # Dependencies
├── dataset/
│   ├── Dictionary.py           # Vocabulary management
│   ├── IterableDataset.py      # Data loading
│   └── __init__.py
├── models/
│   ├── LSTM.py                 # LSTM model
│   ├── Transformer.py          # Transformer model
│   ├── Embedding.py            # Multi-feature embeddings
│   ├── MyMultiheadAttention.py # Custom attention
│   ├── SwiGLU.py               # Activation function
│   ├── LabelSmoothingCrossEntropy.py
│   └── __init__.py
├── scripts/
│   ├── lstm-1.sh               # SLURM training script (LSTM)
│   ├── lstm-2.sh to lstm-5.sh  # Additional LSTM configs
│   ├── transformer-1.sh        # SLURM training script (Transformer)
│   └── transformer-2.sh to transformer-4.sh
└── trace_data.tar.gz           # Dataset (18.9 GB compressed)
```

---

## Key Functions

### Data Processing
- **`load_trace(file_path)`**: Load LTTng trace using babeltrace
- **`get_events(trace, keys)`**: Extract events from trace
- **`get_requests(events)`**: Segment events into requests
- **`generate_dataset_request_based(...)`**: Convert traces to training data
- **`categorize_latency(...)`**: Bin latencies into categories

### Training & Evaluation
- **`train(rank, ...)`**: Multi-GPU training loop (DDP)
- **`evaluate(model, dataset, ...)`**: Compute loss and accuracy
- **`adaptive_tracing_eval(...)`**: OOD detection and root-cause analysis

### N-gram Baseline
- **`nltk_ngram(file_path, n)`**: Build n-gram model
- **`ngram_eval(...)`**: Evaluate n-gram perplexity
- **`ood_detection_ngram(...)`**: N-gram-based anomaly detection

---

## Dependencies

Key libraries:
- **PyTorch**: Deep learning framework
- **babeltrace**: LTTng trace parsing
- **NLTK**: N-gram baseline
- **scikit-learn**: Metrics and clustering
- **matplotlib**: Visualization
- **tqdm**: Progress bars

---

## Research Context & Motivation

### Problem
Traditional system tracing has overhead-observability trade-off:
- **High detail** → High overhead (unusable in production)
- **Low overhead** → Insufficient detail (can't debug issues)

### Solution
**Adaptive tracing**: Start with low overhead, escalate when needed
- Use ML to detect "normal" vs "anomalous" behavior
- Only increase tracing granularity for anomalies
- Provide root-cause hints to operators

### Innovation
Treating kernel events as language:
- System calls have patterns (like sentences)
- Temporal dependencies (like grammar)
- Duration patterns (like prosody)
- Anomalies are "out-of-vocabulary" or "ungrammatical"

---

## Potential Research Extensions

Based on the codebase and reviewer feedback:

### 1. Generalization to Other Systems
- Stream processing (Kafka, Flink)
- Databases (Redis, PostgreSQL)
- Microservices (service meshes)
- Different OS kernels (Windows, macOS)

### 2. Distributed Tracing
- Multi-host event correlation
- Causal ordering across machines
- Distributed anomaly detection
- Cross-service root-cause analysis

### 3. Online Learning
- Continual learning from production data
- Adaptation to workload drift
- Incremental vocabulary updates

### 4. Explainability
- Attention visualization
- Event importance scores
- Counterfactual explanations
- Human-in-the-loop refinement

### 5. Efficiency Improvements
- Model compression (pruning, quantization)
- Faster inference (TensorRT, ONNX)
- Streaming evaluation
- Edge deployment

---

## Summary

LMAT is a **novel approach to system observability** that:
1. Models kernel events as sequences using language models
2. Performs multi-task learning (event + duration prediction)
3. Detects anomalies via out-of-distribution detection
4. Provides interpretable root-cause analysis
5. Enables adaptive tracing (low overhead → high detail on demand)

**Current status**: Strong technical foundation, but needs:
- Broader evaluation (different workload types)
- Performance overhead quantification
- Clearer scope (single-host vs distributed)

**Strengths**:
- Practical model size (~1M params)
- Multi-task learning improves both tasks
- Request-level analysis is intuitive
- Root-cause analysis aids debugging

**Limitations** (per reviewers):
- Only validated on Apache web server
- Unknown impact on application performance
- Unclear generalization to distributed systems
- Limited architectural diversity in evaluation

---

## Next Steps for Research

If building on this work, consider:

1. **Validate on diverse workloads** (addresses Reviewer #3 concern)
2. **Measure end-to-end overhead** (latency, throughput impact)
3. **Explore distributed tracing** (or explicitly scope to single-host)
4. **Investigate model compression** (for lower overhead)
5. **Develop online learning** (adapt to workload changes)
6. **Enhance explainability** (better root-cause attribution)
7. **Compare to recent baselines** (Sieve, TraStrainer, etc.)

The codebase is well-structured and ready for extension. The core ideas are sound, but the evaluation needs strengthening per reviewer feedback.

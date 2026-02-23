#!/usr/bin/env python
"""LMAT dataset preprocessor for Train-Ticket microservice kernel traces.

Unlike the Apache pipeline (which uses ``httpd:enter/exit_event_handler``
as request boundaries), this script uses **per-PID fixed time windows**
to segment the continuous kernel event stream into model-ready sequences.

Data layout
-----------
Normal trace    → chronologically split into train / valid_id / test_id shards.
Anomaly traces  → each becomes one OOD shard directory.

Shared artefacts (compatible with Apache pipeline):
    {output_dir}/dict_sys.pkl
    {output_dir}/dict_proc.pkl
    {output_dir}/delay_spans.pkl

NPZ shard format is identical to the Apache pipeline so ``NPZIterableDataset``
and the training loop require zero changes.

Usage
-----
::

    python data-preprocessing/preprocess_trainticket.py \\
        --normal_trace  /scratch/.../train-ticket-normal-full/kernel \\
        --anomaly_traces \\
            "bandwidth:/scratch/.../train-ticket-bandwidth/kernel,\\
             cpu_stress:/scratch/.../train-ticket-cpu-stress/kernel,\\
             db_load:/scratch/.../train-ticket-db-load/kernel,\\
             io_stress:/scratch/.../train-ticket-io-stress/kernel,\\
             memory:/scratch/.../train-ticket-memory/kernel,\\
             pod_restart:/scratch/.../train-ticket-pod-restart/kernel,\\
             verbose_log:/scratch/.../train-ticket-verbose-logging/kernel" \\
        --output_dir /scratch/.../processed-train-ticket \\
        --window_ms 500 \\
        --min_events 10 \\
        --train_frac 0.70 \\
        --valid_frac 0.15 \\
        --shard_size 5000 \\
        --num_proc 7
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from multiprocessing import Pool
from time import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from vocabulary               import Dictionary
from trace_reader             import load_trace, get_events
from trace_reader_trainticket import get_sequences_by_pid_window, scan_trace_duration
from latency_utils            import (
    ALL_N_CATS,
    build_all_spans,
    merge_all_spans,
    categorize_batch,
)
# Reuse the shard writer and encoder from preprocess.py unchanged
from preprocess               import encode_request, write_shard

# ---------------------------------------------------------------------------
BABELTRACE_KEYS = {
    "vtid": "tid", "tid": "tid",
    "vpid": "pid", "pid": "pid",
    "procname": "procname", "ret": "ret",
}


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LMAT Train-Ticket kernel trace preprocessor."
    )
    p.add_argument("--normal_trace", required=True,
                   help="Path to the normal (non-anomaly) LTTng kernel trace dir.")
    p.add_argument("--anomaly_traces", required=True,
                   help="Comma-separated 'name:path' pairs for anomaly trace dirs.")
    p.add_argument("--output_dir", required=True,
                   help="Root output directory for NPZ shards and vocab files.")

    # Segmentation
    p.add_argument("--window_ms", type=int, default=500,
                   help="Time window size in milliseconds (default: 500).")
    p.add_argument("--min_events", type=int, default=10,
                   help="Min events per window to include (default: 10).")
    p.add_argument("--filter_procs", default=None,
                   help="Comma-separated procname substrings to include "
                        "(default: all processes).")

    # Train/valid/test split of the normal trace
    p.add_argument("--train_frac", type=float, default=0.70,
                   help="Fraction of the normal trace for training (default 0.70).")
    p.add_argument("--valid_frac", type=float, default=0.15,
                   help="Fraction for in-distribution validation (default 0.15). "
                        "Remainder becomes test_id.")

    # Processing
    p.add_argument("--shard_size", type=int, default=5000,
                   help="Requests per NPZ shard (default: 5000).")
    p.add_argument("--num_proc", type=int, default=None,
                   help="Parallel workers for anomaly traces (default: n_anomaly).")
    p.add_argument("--max_windows", type=int, default=None,
                   help="Cap on windows per split (for quick test runs).")

    args = p.parse_args()

    if not os.path.isdir(args.normal_trace):
        p.error(f"normal_trace not found: {args.normal_trace}")
    for spec in args.anomaly_traces.split(","):
        name, path = spec.strip().split(":", 1)
        if not os.path.isdir(path.strip()):
            p.error(f"Anomaly trace not found ({name}): {path.strip()}")
    if not (0 < args.train_frac < 1 and 0 < args.valid_frac < 1
            and args.train_frac + args.valid_frac < 1):
        p.error("train_frac + valid_frac must be < 1.0")

    return args


# ===========================================================================
# Process one trace → write NPZ shards for multiple splits simultaneously
# ===========================================================================

def process_normal_trace(
    trace_path:   str,
    dict_sys:     Dictionary,
    dict_proc:    Dictionary,
    span_list:    List,
    output_dir:   str,
    window_ns:    int,
    min_events:   int,
    filter_procs: Optional[set],
    train_frac:   float,
    valid_frac:   float,
    shard_size:   int,
    max_windows:  Optional[int],
) -> Tuple[List, int, int, int]:
    """Process the single normal trace.

    Determines train/valid_id/test_id boundaries by scanning the trace once
    to find total duration, then processes and routes windows accordingly.

    Returns:
        (updated_span_list, n_train, n_valid, n_test)
    """
    # ------------------------------------------------------------------
    # Pass 1 — fast scan to find total trace duration
    # ------------------------------------------------------------------
    print("\n  [Pass 1] Scanning trace duration…", file=sys.stderr, flush=True)
    t0 = time()
    first_ts, last_ts, n_ev = scan_trace_duration(trace_path)
    total_ns   = last_ts - first_ts
    elapsed    = timedelta(seconds=round(time() - t0))
    print(
        f"  Duration: {total_ns / 1e9:.1f} s  ({n_ev:,} events)  [{elapsed}]",
        file=sys.stderr,
    )

    # Time boundaries for each split
    train_end_ns = first_ts + int(total_ns * train_frac)
    valid_end_ns = first_ts + int(total_ns * (train_frac + valid_frac))

    split_dirs = {
        "train":    os.path.join(output_dir, "train",    "npz"),
        "valid_id": os.path.join(output_dir, "valid_id", "npz"),
        "test_id":  os.path.join(output_dir, "test_id",  "npz"),
    }
    for d in split_dirs.values():
        os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------------
    # Pass 2 — process windows, route to correct split
    # ------------------------------------------------------------------
    print("\n  [Pass 2] Processing windows…", file=sys.stderr, flush=True)

    # Per-split accumulators
    enc_bufs:  Dict[str, List] = {k: [] for k in split_dirs}
    lat_bufs:  Dict[str, List] = {k: [] for k in split_dirs}
    name_bufs: Dict[str, List] = {k: [] for k in split_dirs}
    shard_idx: Dict[str, int]  = {k: 0  for k in split_dirs}

    def _split_name(window_start_ns: int) -> str:
        if window_start_ns < train_end_ns:
            return "train"
        if window_start_ns < valid_end_ns:
            return "valid_id"
        return "test_id"

    trace    = load_trace(trace_path)
    gen      = get_sequences_by_pid_window(
        get_events(trace, BABELTRACE_KEYS),
        window_ns=window_ns,
        min_events=min_events,
        filter_procs=filter_procs,
    )

    total_windows = 0
    time_hist: Optional[Dict] = None
    t0 = time()

    for window_events, procname, pid, win_start_ns in gen:
        if max_windows is not None and total_windows >= max_windows:
            break
        total_windows += 1

        if total_windows % 2000 == 0:
            elapsed = timedelta(seconds=round(time() - t0))
            n_tr = shard_idx["train"] * shard_size + len(enc_bufs["train"])
            n_va = shard_idx["valid_id"] * shard_size + len(enc_bufs["valid_id"])
            n_te = shard_idx["test_id"] * shard_size + len(enc_bufs["test_id"])
            print(
                f"\r  windows={total_windows:>8,}  "
                f"train={n_tr:>6,}  valid={n_va:>6,}  test={n_te:>6,}  "
                f"elapsed={elapsed}",
                end="", file=sys.stderr, flush=True,
            )

        split = _split_name(win_start_ns)
        is_train = split == "train"

        enc = encode_request(window_events, dict_sys, dict_proc, train=is_train)
        if enc is None:
            continue

        enc_bufs[split].append(enc)
        lat_bufs[split].append(enc["latency_ns"])
        name_bufs[split].append(enc["event_names"])

        # Flush when buffer full
        if len(enc_bufs[split]) >= shard_size:
            if is_train and time_hist is not None:
                span_list = merge_all_spans(span_list, time_hist)
                # Trim history to bound memory
                for k in time_hist:
                    time_hist[k] = time_hist[k][-20:]

            lat_cats = categorize_batch(lat_bufs[split], name_bufs[split], span_list)
            write_shard(split_dirs[split], shard_idx[split], enc_bufs[split], lat_cats)
            shard_idx[split] += 1
            enc_bufs[split]  = []
            lat_bufs[split]  = []
            name_bufs[split] = []

    # Final partial flushes for all splits
    for split in split_dirs:
        if not enc_bufs[split]:
            continue
        if split == "train" and time_hist is not None:
            span_list = merge_all_spans(span_list, time_hist)
        lat_cats = categorize_batch(lat_bufs[split], name_bufs[split], span_list)
        write_shard(split_dirs[split], shard_idx[split], enc_bufs[split], lat_cats)
        shard_idx[split] += 1

    elapsed = timedelta(seconds=round(time() - t0))
    print(
        f"\n  ✓ Normal trace done in {elapsed}  "
        f"(train={shard_idx['train']} shards, "
        f"valid={shard_idx['valid_id']} shards, "
        f"test={shard_idx['test_id']} shards)",
        file=sys.stderr,
    )
    return span_list, shard_idx["train"], shard_idx["valid_id"], shard_idx["test_id"]


# ===========================================================================
# Process one anomaly trace (worker for multiprocessing.Pool)
# ===========================================================================

def process_anomaly_trace(
    trace_path:   str,
    split_name:   str,
    dict_sys:     Dictionary,
    dict_proc:    Dictionary,
    span_list:    List,
    output_dir:   str,
    window_ns:    int,
    min_events:   int,
    filter_procs: Optional[set],
    shard_size:   int,
    max_windows:  Optional[int],
) -> None:
    shard_dir = os.path.join(output_dir, split_name, "npz")
    os.makedirs(shard_dir, exist_ok=True)

    trace = load_trace(trace_path)
    gen   = get_sequences_by_pid_window(
        get_events(trace, BABELTRACE_KEYS),
        window_ns=window_ns,
        min_events=min_events,
        filter_procs=filter_procs,
    )

    enc_buf, lat_buf, name_buf = [], [], []
    shard_idx  = 0
    total_wins = 0
    t0         = time()

    for window_events, procname, pid, win_start_ns in gen:
        if max_windows is not None and total_wins >= max_windows:
            break
        total_wins += 1

        enc = encode_request(window_events, dict_sys, dict_proc, train=False)
        if enc is None:
            continue

        enc_buf.append(enc)
        lat_buf.append(enc["latency_ns"])
        name_buf.append(enc["event_names"])

        if len(enc_buf) >= shard_size:
            lat_cats = categorize_batch(lat_buf, name_buf, span_list)
            write_shard(shard_dir, shard_idx, enc_buf, lat_cats)
            shard_idx += 1
            enc_buf, lat_buf, name_buf = [], [], []

    if enc_buf:
        lat_cats = categorize_batch(lat_buf, name_buf, span_list)
        write_shard(shard_dir, shard_idx, enc_buf, lat_cats)
        shard_idx += 1

    elapsed = timedelta(seconds=round(time() - t0))
    print(
        f"\n  ✓ {split_name:30s}  {total_wins:>6,} windows → "
        f"{shard_idx} shards  ({elapsed})",
        file=sys.stderr,
    )


def _anomaly_worker(args: Tuple) -> None:
    (trace_path, split_name, dict_sys, dict_proc, span_list,
     output_dir, window_ns, min_events, filter_procs, shard_size, max_windows) = args
    process_anomaly_trace(
        trace_path, split_name, dict_sys, dict_proc, span_list,
        output_dir, window_ns, min_events, filter_procs, shard_size, max_windows,
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    window_ns    = args.window_ms * 1_000_000
    filter_procs = (set(args.filter_procs.split(","))
                    if args.filter_procs else None)

    anomaly_specs: List[Tuple[str, str]] = []
    for spec in args.anomaly_traces.split(","):
        name, path = spec.strip().split(":", 1)
        anomaly_specs.append((name.strip(), path.strip()))

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Normal trace → train / valid_id / test_id
    # ------------------------------------------------------------------
    print(f"\n[1/2] Processing normal trace → train / valid_id / test_id",
          file=sys.stderr)
    dict_sys  = Dictionary()
    dict_proc = Dictionary()
    span_list = [None, None, None, None]

    span_list, n_tr, n_va, n_te = process_normal_trace(
        trace_path   = args.normal_trace,
        dict_sys     = dict_sys,
        dict_proc    = dict_proc,
        span_list    = span_list,
        output_dir   = args.output_dir,
        window_ns    = window_ns,
        min_events   = args.min_events,
        filter_procs = filter_procs,
        train_frac   = args.train_frac,
        valid_frac   = args.valid_frac,
        shard_size   = args.shard_size,
        max_windows  = args.max_windows,
    )

    # Save shared artefacts (same format as Apache pipeline)
    dict_sys.save(os.path.join(args.output_dir, "dict_sys.pkl"))
    dict_proc.save(os.path.join(args.output_dir, "dict_proc.pkl"))
    with open(os.path.join(args.output_dir, "delay_spans.pkl"), "wb") as fh:
        pickle.dump(span_list, fh)

    print(
        f"\n  Vocabulary: {len(dict_sys)} syscalls, {len(dict_proc)} processes",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Step 2 — Anomaly traces → OOD shards (parallel)
    # ------------------------------------------------------------------
    print(f"\n[2/2] Processing {len(anomaly_specs)} anomaly traces in parallel…",
          file=sys.stderr)

    n_workers = min(
        args.num_proc if args.num_proc else os.cpu_count(),
        len(anomaly_specs),
    )

    worker_args = [
        (
            path, name, dict_sys, dict_proc, span_list,
            args.output_dir, window_ns, args.min_events,
            filter_procs, args.shard_size, args.max_windows,
        )
        for name, path in anomaly_specs
    ]

    if n_workers <= 1 or len(anomaly_specs) <= 1:
        for wa in worker_args:
            _anomaly_worker(wa)
    else:
        with Pool(processes=n_workers) as pool:
            pool.map(_anomaly_worker, worker_args)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"  Output:  {args.output_dir}", file=sys.stderr)
    print(f"  Splits:  train({n_tr}sh) valid_id({n_va}sh) test_id({n_te}sh)",
          file=sys.stderr)
    print(f"  OOD:     {[n for n,_ in anomaly_specs]}", file=sys.stderr)
    print(f"  Vocab:   {len(dict_sys)} syscalls × {len(dict_proc)} processes",
          file=sys.stderr)
    print(f"  Window:  {args.window_ms} ms / PID  (min {args.min_events} events)",
          file=sys.stderr)
    print(f"{'='*70}\n✓ Done.", file=sys.stderr)


if __name__ == "__main__":
    main()

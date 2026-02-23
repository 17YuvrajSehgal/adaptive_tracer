#!/usr/bin/env python
"""Standalone LMAT dataset preprocessing.

Reads LTTng binary traces via babeltrace2, builds vocabulary and latency-span
dictionaries from the training split, then processes all other splits (in
parallel) and writes the results as **NPZ shards** for fast, zero-parse
loading at training time.

Output layout
-------------
``{data_path}/``
├── dict_sys.pkl           # syscall-name vocabulary (same pickle as before)
├── dict_proc.pkl          # process-name vocabulary
├── delay_spans.pkl        # list of 4 span dicts (n_cat 4,6,8,10)
└── {split_folder}/
    └── npz/
        ├── shard_0000.npz
        ├── shard_0001.npz
        └── ...

Each NPZ shard contains
-----------------------
call        int32  (N, L)  system-call token ids  (padded with 0)
entry       int8   (N, L)  0=none / 1=entry / 2=exit
duration    int64  (N, L)  nanoseconds since previous event
proc        int32  (N, L)  process-name token ids
pid         int32  (N, L)  process id (sinusoidal-encoded at model time)
tid         int32  (N, L)  thread  id
ret         int8   (N, L)  0=no-ret / 1=success / 2=failure
lat4        int8   (N, L)  latency category (n_categories=4)
lat6        int8   (N, L)  latency category (n_categories=6)
lat8        int8   (N, L)  latency category (n_categories=8)
lat10       int8   (N, L)  latency category (n_categories=10)
lengths     int32  (N,)    true sequence length (to build pad mask)
timestamps  int64  (N,)    nanosecond timestamp of first event in request
req_dur     float32(N,)    total request duration in ms

where N = number of requests in the shard (≤ SHARD_SIZE) and L = the
maximum sequence length **within that shard** (varies across shards).

Speedups vs. original ``generate_dataset_request_based``
---------------------------------------------------------
1. Vectorised latency categorisation via ``np.searchsorted`` instead of a
   Python inner loop.
2. OOD splits processed in parallel via ``multiprocessing.Pool``.
3. NPZ output: training reads raw numpy arrays; no semicolon-parsing.
4. Buffered shard writes (one ``np.savez`` per SHARD_SIZE requests) instead
   of one ``f.write`` per field per request.

Usage
-----
::

    python data-preprocessing/preprocess.py \\
        --data_path /scratch/.../trace_data \\
        --train_folder      "Train:train_id" \\
        --valid_id_folder   "Valid ID:valid_id" \\
        --valid_ood_folders "Valid OOD (Connection):valid_ood_connection,..." \\
        --test_id_folder    "Test ID:test_id" \\
        --test_ood_folders  "Test OOD (Connection):test_ood_connection,..." \\
        --shard_size 5000 \\
        --num_proc 8

"""

from __future__ import annotations

import argparse
import itertools
import os
import pickle
import sys
from multiprocessing import Pool
from time import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Allow running from repo root: ``python data-preprocessing/preprocess.py``
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from vocabulary    import Dictionary
from trace_reader  import load_trace, get_events, get_requests
from latency_utils import (
    ALL_N_CATS,
    build_all_spans,
    merge_all_spans,
    categorize_batch,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SHARD_SIZE = 5000   # requests per NPZ shard
SKIP_FIRST_N       = 1000   # discard first 1 000 requests (trace warm-up)
SAVE_THRESH_FIRST  = 50_000 # for train: first batch threshold before flushing
SAVE_THRESH_REST   = 10_000 # subsequent batches

BABELTRACE_KEYS = {
    "vtid":     "tid",
    "tid":      "tid",
    "vpid":     "pid",
    "pid":      "pid",
    "procname": "procname",
    "ret":      "ret",
}

LAT_NAMES = {4: "lat4", 6: "lat6", 8: "lat8", 10: "lat10"}


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LMAT standalone dataset preprocessor — writes NPZ shards."
    )

    # Paths (same naming as main.py so scripts can share flags)
    p.add_argument("--data_path",        required=True,
                   help="Root folder containing all split sub-folders.")
    p.add_argument("--train_folder",     required=True,
                   help="'Display Name:folder' for the training split.")
    p.add_argument("--valid_id_folder",  required=True,
                   help="'Display Name:folder' for the in-distribution validation split.")
    p.add_argument("--valid_ood_folders", required=True,
                   help="Comma-separated 'Name:folder' pairs for OOD validation splits.")
    p.add_argument("--test_id_folder",   required=True,
                   help="'Display Name:folder' for the in-distribution test split.")
    p.add_argument("--test_ood_folders", required=True,
                   help="Comma-separated 'Name:folder' pairs for OOD test splits.")

    # Processing options
    p.add_argument("--shard_size", type=int, default=DEFAULT_SHARD_SIZE,
                   help=f"Requests per NPZ shard (default: {DEFAULT_SHARD_SIZE}).")
    p.add_argument("--num_proc",   type=int, default=None,
                   help="Parallel worker processes for OOD splits "
                        "(default: min(n_ood_splits, os.cpu_count())).")
    p.add_argument("--max_sample", type=int, default=None,
                   help="Maximum number of requests to read per split "
                        "(useful for small test runs).")

    args = p.parse_args()

    # Validate that folders exist
    def _check(folder_spec: str) -> None:
        folder = folder_spec.split(":")[1]
        path   = os.path.join(args.data_path, folder)
        if not os.path.isdir(path):
            p.error(f"Folder not found: {path}")

    _check(args.train_folder)
    _check(args.valid_id_folder)
    _check(args.test_id_folder)
    for spec in args.valid_ood_folders.split(","):
        _check(spec)
    for spec in args.test_ood_folders.split(","):
        _check(spec)

    return args


# ===========================================================================
# Core: encode a single request into integer arrays
# ===========================================================================

def encode_request(
    request:   List[Dict],
    dict_sys:  Dictionary,
    dict_proc: Dictionary,
    train:     bool,
) -> Optional[Dict]:
    """Convert one raw request (list of event dicts) into integer sequences.

    Args:
        request:   Ordered list of event dicts from ``get_requests``.
        dict_sys:  System-call vocabulary.
        dict_proc: Process-name vocabulary.
        train:     If True, add unseen tokens to the vocabularies.

    Returns:
        A dict with keys call / entry / duration / proc / pid / tid / ret /
        latency_ns / event_names / timestamp_ns / req_duration_ms.
        Returns ``None`` for empty requests.
    """
    if not request:
        return None

    call  = [dict_sys.get_idx("[START]")]
    proc  = [dict_proc.get_idx("[START]")]
    entry    = [0]
    duration = [0]
    pid      = [0]
    tid      = [0]
    ret      = [0]
    latency_ns  = [0]   # raw ns latency per event (for categorisation)
    event_names = [""]  # full event name (for categorize_batch)
    timestamp_ns: Optional[int] = None
    prev_ts: Optional[int]      = None

    for event in request:
        name     = event["name"]
        sysname  = name.replace("syscall_", "").replace("entry_", "").replace("exit_", "")
        procname = str(event.get("procname", ""))

        if train:
            dict_sys.add_word(sysname)
            dict_proc.add_word(procname)

        call.append(dict_sys.get_idx(sysname))

        # entry / exit flag
        if "entry" in name:
            entry.append(1)
        elif "exit" in name:
            entry.append(2)
        else:
            entry.append(0)

        # inter-event elapsed time (ns)
        cur_ts = event["timestamp"]
        if prev_ts is not None:
            duration.append(cur_ts - prev_ts)
        else:
            duration.append(0)
        prev_ts = cur_ts

        if timestamp_ns is None:
            timestamp_ns = cur_ts

        proc.append(dict_proc.get_idx(procname))
        pid.append(event.get("pid", 0))
        tid.append(event.get("tid", 0))

        # return value encoding
        if "entry" in name:
            ret.append(0)
        else:
            rv = event.get("ret", None)
            if rv is None:
                ret.append(0)
            elif rv >= 0:
                ret.append(1)
            else:
                ret.append(2)

        latency_ns.append(event.get("latency") or 0)
        event_names.append(name)

    # [END] token
    call.append(dict_sys.get_idx("[END]"))
    proc.append(dict_proc.get_idx("[END]"))
    entry.append(0);    duration.append(0)
    pid.append(0);      tid.append(0);   ret.append(0)
    latency_ns.append(0);    event_names.append("")

    req_dur_ms = sum(duration) / 1e6

    return {
        "call":         call,
        "entry":        entry,
        "duration":     duration,
        "proc":         proc,
        "pid":          pid,
        "tid":          tid,
        "ret":          ret,
        "latency_ns":   latency_ns,
        "event_names":  event_names,
        "timestamp_ns": timestamp_ns or 0,
        "req_dur_ms":   req_dur_ms,
    }


# ===========================================================================
# NPZ shard writer
# ===========================================================================

def write_shard(
    shard_dir:   str,
    shard_index: int,
    encoded:     List[Dict],
    lat_cats:    Dict[int, List[List[int]]],
) -> None:
    """Pad a batch of encoded requests and write one NPZ shard.

    Args:
        shard_dir:   Directory to write the shard into (created if needed).
        shard_index: Zero-based shard number (used in filename).
        encoded:     List of dicts from ``encode_request``.
        lat_cats:    Dict from ``categorize_batch``, keyed by n_cat integer.
    """
    os.makedirs(shard_dir, exist_ok=True)

    N   = len(encoded)
    L   = max(len(r["call"]) for r in encoded)

    # Allocate padded arrays (int32 / int64 / float32)
    call     = np.zeros((N, L), dtype=np.int32)
    entry    = np.zeros((N, L), dtype=np.int8)
    duration = np.zeros((N, L), dtype=np.int64)
    proc     = np.zeros((N, L), dtype=np.int32)
    pid_arr  = np.zeros((N, L), dtype=np.int32)
    tid_arr  = np.zeros((N, L), dtype=np.int32)
    ret      = np.zeros((N, L), dtype=np.int8)
    lat4     = np.zeros((N, L), dtype=np.int8)
    lat6     = np.zeros((N, L), dtype=np.int8)
    lat8     = np.zeros((N, L), dtype=np.int8)
    lat10    = np.zeros((N, L), dtype=np.int8)
    lengths    = np.zeros(N, dtype=np.int32)
    timestamps = np.zeros(N, dtype=np.int64)
    req_dur    = np.zeros(N, dtype=np.float32)

    for i, r in enumerate(encoded):
        seq_len = len(r["call"])
        lengths[i]      = seq_len
        timestamps[i]   = r["timestamp_ns"]
        req_dur[i]      = r["req_dur_ms"]

        call[i,     :seq_len] = r["call"]
        entry[i,    :seq_len] = r["entry"]
        duration[i, :seq_len] = r["duration"]
        proc[i,     :seq_len] = r["proc"]
        pid_arr[i,  :seq_len] = r["pid"]
        tid_arr[i,  :seq_len] = r["tid"]
        ret[i,      :seq_len] = r["ret"]

    # Clamp lat values to int8 range [-128, 127] — category ids are small
    for i, cats_per_req in enumerate(lat_cats[4]):
        seq_len = lengths[i]
        lat4[i, :seq_len] = np.clip(cats_per_req[:seq_len], -128, 127)
    for i, cats_per_req in enumerate(lat_cats[6]):
        seq_len = lengths[i]
        lat6[i, :seq_len] = np.clip(cats_per_req[:seq_len], -128, 127)
    for i, cats_per_req in enumerate(lat_cats[8]):
        seq_len = lengths[i]
        lat8[i, :seq_len] = np.clip(cats_per_req[:seq_len], -128, 127)
    for i, cats_per_req in enumerate(lat_cats[10]):
        seq_len = lengths[i]
        lat10[i, :seq_len] = np.clip(cats_per_req[:seq_len], -128, 127)

    shard_path = os.path.join(shard_dir, f"shard_{shard_index:04d}.npz")
    np.savez(
        shard_path,
        call=call, entry=entry, duration=duration,
        proc=proc, pid=pid_arr, tid=tid_arr, ret=ret,
        lat4=lat4, lat6=lat6, lat8=lat8, lat10=lat10,
        lengths=lengths, timestamps=timestamps, req_dur=req_dur,
    )


# ===========================================================================
# Processing one split
# ===========================================================================

def process_split(
    split_path:   str,
    dict_sys:     Dictionary,
    dict_proc:    Dictionary,
    span_list:    List[Optional[Dict]],
    train:        bool,
    shard_size:   int,
    max_sample:   Optional[int],
) -> List[Optional[Dict]]:
    """Read a trace split and write NPZ shards.

    For the training split (``train=True``), the span dictionaries are built
    incrementally and returned.  For other splits the spans are fixed (as
    computed from training) and returned unchanged.

    Args:
        split_path:  Absolute path to the trace folder (contains CTF files).
        dict_sys:    System-call vocabulary (updated in-place when ``train``).
        dict_proc:   Process-name vocabulary (updated in-place when ``train``).
        span_list:   Four span dicts (one per n_cat).  ``None`` entries mean
                     "not yet built" (only during first training batch).
        train:       Whether to update vocabularies and span dicts.
        shard_size:  Requests per NPZ shard file.
        max_sample:  Cap on total requests to process (``None`` = unlimited).

    Returns:
        (Potentially updated) span_list.
    """
    shard_dir = os.path.join(split_path, "npz")
    os.makedirs(shard_dir, exist_ok=True)

    trace = load_trace(split_path)
    gen   = itertools.islice(
        get_requests(get_events(trace, BABELTRACE_KEYS)),
        SKIP_FIRST_N, None,
    )

    # For early batches in training we accumulate more before flushing
    # (mirrors the original SAVE_THRESH_FIRST / SAVE_THRESH_REST logic)
    flush_after   = SAVE_THRESH_FIRST if train else shard_size
    first_flush   = True

    encoded_buf:    List[Dict]           = []
    lat_ns_buf:     List[List[int]]      = []
    ev_names_buf:   List[List[str]]      = []
    time_hist:      Optional[Dict]       = None

    shard_idx = 0
    total_req = 0
    t0        = time()

    for i, (request, tmp_hist, _err) in enumerate(gen):
        if max_sample is not None and total_req >= max_sample:
            break

        # Progress
        if i % 5000 == 0:
            elapsed = timedelta(seconds=round(time() - t0))
            print(
                f"\r  {split_path:50s}  req {total_req:>8,}  "
                f"shards {shard_idx:>4}  elapsed {elapsed}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        # Keep history (for span updates on the training set)
        time_hist = tmp_hist

        enc = encode_request(request, dict_sys, dict_proc, train)
        if enc is None:
            continue

        encoded_buf.append(enc)
        lat_ns_buf.append(enc["latency_ns"])
        ev_names_buf.append(enc["event_names"])
        total_req += 1

        # Flush a shard when buffer is full
        if len(encoded_buf) >= flush_after:

            # Update spans from accumulated history (training only)
            if train and time_hist:
                span_list = merge_all_spans(span_list, time_hist)
                # Trim history to last 20 entries per event (memory management)
                for k in time_hist:
                    time_hist[k] = time_hist[k][-20:]

            # Categorise latencies with current spans
            lat_cats = categorize_batch(lat_ns_buf, ev_names_buf, span_list)

            # Split buffer into shards of exactly shard_size
            for start in range(0, len(encoded_buf), shard_size):
                batch      = encoded_buf[start: start + shard_size]
                batch_cats = {
                    nc: lat_cats[nc][start: start + shard_size]
                    for nc in ALL_N_CATS
                }
                write_shard(shard_dir, shard_idx, batch, batch_cats)
                shard_idx += 1

            encoded_buf  = []
            lat_ns_buf   = []
            ev_names_buf = []

            if first_flush and train:
                first_flush = False
                flush_after = shard_size  # switch to rolling shard size

    # Flush remaining requests
    if encoded_buf:
        if train and time_hist:
            span_list = merge_all_spans(span_list, time_hist)
        lat_cats = categorize_batch(lat_ns_buf, ev_names_buf, span_list)
        for start in range(0, len(encoded_buf), shard_size):
            batch      = encoded_buf[start: start + shard_size]
            batch_cats = {nc: lat_cats[nc][start: start + shard_size] for nc in ALL_N_CATS}
            write_shard(shard_dir, shard_idx, batch, batch_cats)
            shard_idx += 1

    elapsed = timedelta(seconds=round(time() - t0))
    print(
        f"\n  ✓ {split_path:50s}  {total_req:>8,} requests → "
        f"{shard_idx} shards  ({elapsed})",
        file=sys.stderr,
    )

    return span_list


# ===========================================================================
# Worker shim for multiprocessing (must be at module level)
# ===========================================================================

def _ood_worker(args: Tuple) -> None:
    """Subprocess entry point for processing one OOD split.

    Receives frozen vocab and span dicts via pickle (standard multiprocessing
    mechanism).  Does NOT modify dict_sys / dict_proc / span_list.
    """
    (
        split_path,
        dict_sys_pkl,
        dict_proc_pkl,
        span_list_pkl,
        shard_size,
        max_sample,
    ) = args

    # Deserialise (already done by pickle; just unpack)
    dict_sys  = dict_sys_pkl
    dict_proc = dict_proc_pkl
    span_list = span_list_pkl

    process_split(
        split_path=split_path,
        dict_sys=dict_sys,
        dict_proc=dict_proc,
        span_list=span_list,
        train=False,
        shard_size=shard_size,
        max_sample=max_sample,
    )


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Build the list of all splits
    # ------------------------------------------------------------------
    def _split_spec(spec: str) -> Tuple[str, str]:
        name, folder = spec.split(":", 1)
        return name.strip(), os.path.join(args.data_path, folder.strip())

    train_name, train_path = _split_spec(args.train_folder)
    vid_name,   vid_path   = _split_spec(args.valid_id_folder)
    tid_name,   tid_path   = _split_spec(args.test_id_folder)

    ood_specs: List[Tuple[str, str]] = []
    for spec in args.valid_ood_folders.split(","):
        ood_specs.append(_split_spec(spec.strip()))
    for spec in args.test_ood_folders.split(","):
        ood_specs.append(_split_spec(spec.strip()))
    # Also add valid_id and test_id to the non-parallel queue
    non_train_specs: List[Tuple[str, str]] = [(vid_name, vid_path), (tid_name, tid_path)] + ood_specs

    # ------------------------------------------------------------------
    # Step 1 — Process training split (builds vocab + spans)
    # ------------------------------------------------------------------
    print(f"\n[1/3] Processing training split: {train_name}", file=sys.stderr)
    dict_sys  = Dictionary()
    dict_proc = Dictionary()
    span_list = [None, None, None, None]

    span_list = process_split(
        split_path=train_path,
        dict_sys=dict_sys,
        dict_proc=dict_proc,
        span_list=span_list,
        train=True,
        shard_size=args.shard_size,
        max_sample=args.max_sample,
    )

    # Save shared artefacts (same filenames as original main.py)
    dict_sys.save(os.path.join(args.data_path, "dict_sys.pkl"))
    dict_proc.save(os.path.join(args.data_path, "dict_proc.pkl"))
    with open(os.path.join(args.data_path, "delay_spans.pkl"), "wb") as fh:
        pickle.dump(span_list, fh)

    print(
        f"\n  Vocabulary: {len(dict_sys)} syscalls, {len(dict_proc)} processes",
        file=sys.stderr,
    )

    # ------------------------------------------------------------------
    # Step 2 — Process Valid-ID and Test-ID (sequential, small)
    # ------------------------------------------------------------------
    print(f"\n[2/3] Processing ID splits: {vid_name}, {tid_name}", file=sys.stderr)
    for _, path in [(vid_name, vid_path), (tid_name, tid_path)]:
        process_split(
            split_path=path,
            dict_sys=dict_sys,
            dict_proc=dict_proc,
            span_list=span_list,
            train=False,
            shard_size=args.shard_size,
            max_sample=args.max_sample,
        )

    # ------------------------------------------------------------------
    # Step 3 — Process OOD splits in parallel
    # ------------------------------------------------------------------
    print(f"\n[3/3] Processing {len(ood_specs)} OOD splits in parallel…",
          file=sys.stderr)

    n_workers = min(
        args.num_proc if args.num_proc else os.cpu_count(),
        len(ood_specs),
    )

    worker_args = [
        (
            path,
            dict_sys,    # pickle-friendly
            dict_proc,
            span_list,
            args.shard_size,
            args.max_sample,
        )
        for _, path in ood_specs
    ]

    if n_workers <= 1 or len(ood_specs) <= 1:
        for wa in worker_args:
            _ood_worker(wa)
    else:
        with Pool(processes=n_workers) as pool:
            pool.map(_ood_worker, worker_args)

    print("\n✓ All splits preprocessed.", file=sys.stderr)


if __name__ == "__main__":
    main()

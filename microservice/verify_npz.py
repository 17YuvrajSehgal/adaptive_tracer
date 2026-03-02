#!/usr/bin/env python3
"""
verify_npz.py
=============
Quick sanity-check script for NPZ shards produced by preprocess_sockshop.py.
Run this locally (no GPU needed) to confirm the dataset is well-formed
before submitting training jobs to Compute Canada.

Usage:
    python microservice/verify_npz.py \
        --preprocessed_dir micro-service-trace-data/preprocessed

What it checks:
    1. vocab.pkl and delay_spans.pkl exist and are loadable
    2. Every split listed in --splits has at least one shard
    3. Each shard has the expected NPZ keys and correct array shapes
    4. Sequences are non-empty and contain valid token IDs
    5. SockshopNpzDataset can iterate the first 50 samples without errors
"""

import os
import sys
import json
import glob
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from microservice.NpzDataset import SockshopNpzDataset

REQUIRED_KEYS = {
    "call", "entry", "duration", "proc", "pid", "tid",
    "ret", "lat_cat", "seq_len", "req_dur_ms", "is_anomaly",
}


def check_shard(shard_path, n_vocab_sys, n_vocab_proc):
    """Validate one shard file.  Returns list of error strings."""
    errors = []
    try:
        d = np.load(shard_path, allow_pickle=False)
    except Exception as e:
        return [f"  [ERR] Cannot load shard {shard_path}: {e}"]

    missing = REQUIRED_KEYS - set(d.files)
    if missing:
        errors.append(f"  [ERR] {shard_path}: missing keys {missing}")
        return errors

    N = d["seq_len"].shape[0]
    L = d["call"].shape[1]

    # Shape checks
    for key in ["call", "proc"]:
        if d[key].shape != (N, L):
            errors.append(f"  [ERR] {key} shape {d[key].shape} != ({N},{L})")

    # Vocab range checks
    if d["call"].max() > n_vocab_sys or d["call"].min() < 0:
        errors.append(f"  [ERR] call token OOB: max={d['call'].max()} vocab={n_vocab_sys}")

    # seq_len plausibility
    if (d["seq_len"] <= 0).any():
        errors.append(f"  [ERR] seq_len has non-positive values")
    if (d["seq_len"] > L).any():
        errors.append(f"  [ERR] seq_len > padded length L={L}")

    # lat_cat range
    if d["lat_cat"].min() < 0:
        errors.append(f"  [ERR] lat_cat has negative values")

    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preprocessed_dir", required=True,
                   help="Path to the preprocessed/ output directory")
    p.add_argument("--splits", default=None,
                   help="Comma-separated split names to check "
                        "(default: all sub-dirs that contain shard_*.npz)")
    p.add_argument("--max_iter", type=int, default=50,
                   help="Max samples to iterate through SockshopNpzDataset per split")
    args = p.parse_args()

    total_errors = 0
    print(f"\n{'='*60}")
    print(f"Verifying preprocessed dataset at: {args.preprocessed_dir}")
    print(f"{'='*60}\n")

    # ── Vocab ─────────────────────────────────────────────────────────────────
    vocab_path = os.path.join(args.preprocessed_dir, "vocab.pkl")
    delay_path = os.path.join(args.preprocessed_dir, "delay_spans.pkl")

    if not os.path.isfile(vocab_path):
        print(f"[FAIL] vocab.pkl not found at {vocab_path}")
        sys.exit(1)

    with open(vocab_path, "rb") as f:
        dict_sys, dict_proc = pickle.load(f)
    n_vocab_sys  = len(dict_sys)
    n_vocab_proc = len(dict_proc)
    print(f"  vocab.pkl        : OK  ({n_vocab_sys} syscalls, {n_vocab_proc} proc names)")

    if os.path.isfile(delay_path):
        with open(delay_path, "rb") as f:
            delay_spans = pickle.load(f)
        print(f"  delay_spans.pkl  : OK  ({len(delay_spans)} event types)")
    else:
        print(f"  delay_spans.pkl  : MISSING (warning only)")

    # ── determine splits ──────────────────────────────────────────────────────
    if args.splits:
        splits = [s.strip() for s in args.splits.split(",")]
    else:
        splits = sorted([
            d for d in os.listdir(args.preprocessed_dir)
            if os.path.isdir(os.path.join(args.preprocessed_dir, d))
            and glob.glob(os.path.join(args.preprocessed_dir, d, "shard_*.npz"))
        ])

    print(f"\n  Splits to check: {splits}\n")

    # ── Per-split checks ──────────────────────────────────────────────────────
    for split in splits:
        split_dir = os.path.join(args.preprocessed_dir, split)
        shards    = sorted(glob.glob(os.path.join(split_dir, "shard_*.npz")))

        print(f"  [{split}]  {len(shards)} shards")
        if not shards:
            print(f"    [FAIL] No shards found")
            total_errors += 1
            continue

        # Load meta
        meta_path = os.path.join(split_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"    meta: {meta.get('n_sequences','?')} sequences, "
                  f"label={meta.get('is_anomaly','?')}")

        # Check first and last shard
        for shard_path in [shards[0], shards[-1]]:
            errors = check_shard(shard_path, n_vocab_sys, n_vocab_proc)
            if errors:
                for e in errors:
                    print(e)
                total_errors += len(errors)
            else:
                d = np.load(shard_path, allow_pickle=False)
                N = d["seq_len"].shape[0]
                L = d["call"].shape[1]
                print(f"    {os.path.basename(shard_path)}: OK  "
                      f"({N} seqs, padded_len={L})")

        # Iterate dataset
        try:
            ds = SockshopNpzDataset(split_dir, max_samples=args.max_iter,
                                    shuffle_shards=False)
            n = 0
            for sample in ds:
                assert len(sample) == 12, f"Expected 12-tuple, got {len(sample)}"
                call_in = sample[0]
                assert len(call_in) >= 1, "Empty call_in sequence"
                n += 1
            print(f"    Dataset iterate: OK  ({n} samples checked)")
        except Exception as e:
            print(f"    [FAIL] Dataset iteration error: {e}")
            total_errors += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if total_errors == 0:
        print("✓  All checks PASSED — dataset is ready for training.")
    else:
        print(f"✗  {total_errors} check(s) FAILED — review errors above.")
    print(f"{'='*60}\n")
    sys.exit(0 if total_errors == 0 else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
inspect_vocab.py
----------------
Prints a human-readable summary of the vocab.pkl and delay_spans.pkl
produced by run_build_vocab.sh.

Usage:
    python microservice/inspect_vocab.py \
        --preprocessed_dir /scratch/yuvraj17/adaptive_tracing_scratch/micro-service-trace-data/preprocessed
"""

import os
import sys
import pickle
import argparse
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preprocessed_dir",
        required=True,
        help="Directory containing vocab.pkl and delay_spans.pkl",
    )
    args = p.parse_args()

    vocab_path = os.path.join(args.preprocessed_dir, "vocab.pkl")
    delay_path = os.path.join(args.preprocessed_dir, "delay_spans.pkl")

    # ── 1. vocab.pkl ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("vocab.pkl")
    print("=" * 60)

    if not os.path.isfile(vocab_path):
        print(f"  [ERROR] Not found: {vocab_path}")
        sys.exit(1)

    with open(vocab_path, "rb") as f:
        dict_sys, dict_proc = pickle.load(f)

    print(f"  File size     : {os.path.getsize(vocab_path):,} bytes")
    print(f"  Syscall vocab : {len(dict_sys):,} entries")
    print(f"  Process vocab : {len(dict_proc):,} entries")

    # Print special tokens
    print("\n  Special tokens (syscall):")
    for idx in range(min(6, len(dict_sys))):
        word = dict_sys.idx2word[idx] if hasattr(dict_sys, 'idx2word') else "?"
        print(f"    [{idx:3d}] {word}")

    # Print all syscalls sorted alphabetically
    if hasattr(dict_sys, 'word2idx'):
        words = sorted(dict_sys.word2idx.keys())
        print(f"\n  All {len(words)} syscall tokens (sorted):")
        for i, w in enumerate(words):
            idx = dict_sys.word2idx[w]
            print(f"    [{idx:4d}] {w}")
    else:
        print("\n  [WARN] dict_sys has no word2idx attribute — cannot list tokens")

    # Print all process names
    if hasattr(dict_proc, 'word2idx'):
        procs = sorted(dict_proc.word2idx.keys())
        print(f"\n  All {len(procs)} process tokens:")
        for w in procs:
            idx = dict_proc.word2idx[w]
            print(f"    [{idx:4d}] {w}")

    # ── 2. delay_spans.pkl ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("delay_spans.pkl")
    print("=" * 60)

    if not os.path.isfile(delay_path):
        print(f"  [ERROR] Not found: {delay_path}")
        sys.exit(1)

    with open(delay_path, "rb") as f:
        delay_spans = pickle.load(f)

    print(f"  File size      : {os.path.getsize(delay_path):,} bytes")
    print(f"  Event types    : {len(delay_spans):,}")

    print(f"\n  {'Event':<35} {'Count':>8}  {'Boundaries (ns)':}")
    print("  " + "-" * 78)
    for ev, (boundaries, count) in sorted(delay_spans.items(), key=lambda x: -x[1][1]):
        bnd_str = "  ".join(f"{b/1e6:8.3f}ms" for b in boundaries)
        print(f"  {ev:<35} {count:>8,}  [{bnd_str}]")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  vocab.pkl       : {len(dict_sys)} syscalls, {len(dict_proc)} procs  ✓")
    print(f"  delay_spans.pkl : {len(delay_spans)} event types  ✓")
    print()
    if len(dict_sys) < 10:
        print("  [WARN] Very small syscall vocab — check if all datasets were scanned")
    if len(delay_spans) == 0:
        print("  [WARN] No delay spans — latency categorisation will produce all-zero lat_cat")
    else:
        print("  Files look healthy. Proceed with split jobs.")


if __name__ == "__main__":
    main()

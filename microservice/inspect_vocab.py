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


def _ensure_project_root_on_syspath() -> None:
    """
    Ensure the repository root (the directory that contains the `dataset/` package)
    is on sys.path.

    This is required because vocab.pkl contains pickled instances of classes from
    the `dataset` module, and unpickling requires importing that module.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))  # .../adaptive_tracer
    dataset_dir = os.path.join(repo_root, "dataset")

    if os.path.isdir(dataset_dir) and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


class _RemappingUnpickler(pickle.Unpickler):
    """
    A safer unpickler that can remap module names if the pickle was produced under
    a different import path. This keeps the fix robust across different run layouts.
    """

    _MODULE_REMAP = {
        # common remaps if the project was executed as a package
        "adaptive_tracer.dataset": "dataset",
        "microservice.dataset": "dataset",
    }

    def find_class(self, module, name):
        module = self._MODULE_REMAP.get(module, module)
        return super().find_class(module, name)


def _load_pickle(path: str):
    """
    Load a pickle with a helpful error message if it references project-local modules.
    """
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            # Retry with remapping unpickler (helps when module paths differ)
            f.seek(0)
            try:
                return _RemappingUnpickler(f).load()
            except Exception:
                raise ModuleNotFoundError(
                    f"{e}\n\n"
                    f"While unpickling: {path}\n"
                    f"This pickle likely contains objects from the project's local `dataset` package.\n"
                    f"Fix: run this script from the repository root OR ensure the repo root is on PYTHONPATH.\n"
                    f"Example:\n"
                    f"  python microservice/inspect_vocab.py --preprocessed_dir <dir>\n"
                ) from e


def main():
    _ensure_project_root_on_syspath()

    # Ensure `dataset` can be imported before unpickling (for a clearer error)
    try:
        import dataset  # noqa: F401
    except ModuleNotFoundError as e:
        here = os.path.abspath(os.path.dirname(__file__))
        repo_root = os.path.abspath(os.path.join(here, os.pardir))
        raise SystemExit(
            "Cannot import the local `dataset` package, which is required to unpickle vocab.pkl.\n"
            f"Tried to add repo root to sys.path: {repo_root}\n"
            "If you moved files around, make sure there is a `dataset/` folder at the repo root.\n"
        ) from e

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

    dict_sys, dict_proc = _load_pickle(vocab_path)

    print(f"  File size     : {os.path.getsize(vocab_path):,} bytes")
    print(f"  Syscall vocab : {len(dict_sys):,} entries")
    print(f"  Process vocab : {len(dict_proc):,} entries")

    # Print special tokens
    print("\n  Special tokens (syscall):")
    for idx in range(min(6, len(dict_sys))):
        word = dict_sys.idx2word[idx] if hasattr(dict_sys, "idx2word") else "?"
        print(f"    [{idx:3d}] {word}")

    # Print all syscalls sorted alphabetically
    if hasattr(dict_sys, "word2idx"):
        words = sorted(dict_sys.word2idx.keys())
        print(f"\n  All {len(words)} syscall tokens (sorted):")
        for w in words:
            idx = dict_sys.word2idx[w]
            print(f"    [{idx:4d}] {w}")
    else:
        print("\n  [WARN] dict_sys has no word2idx attribute — cannot list tokens")

    # Print all process names
    if hasattr(dict_proc, "word2idx"):
        procs = sorted(dict_proc.word2idx.keys())
        print(f"\n  All {len(procs)} process tokens:")
        for w in procs:
            idx = dict_proc.word2idx[w]
            print(f"    [{idx:4d}] {w}")
    else:
        print("\n  [WARN] dict_proc has no word2idx attribute — cannot list tokens")

    # ── 2. delay_spans.pkl ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("delay_spans.pkl")
    print("=" * 60)

    if not os.path.isfile(delay_path):
        print(f"  [ERROR] Not found: {delay_path}")
        sys.exit(1)

    delay_spans = _load_pickle(delay_path)

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
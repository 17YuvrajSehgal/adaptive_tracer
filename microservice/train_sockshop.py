#!/usr/bin/env python3
"""
train_sockshop.py
=================

Training entry point for LMAT models on the SockShop NPZ dataset.

This script mirrors the training logic in main.py but uses the fast
SockshopNpzDataset (NPZ shards) instead of the original text-based
IterableDataset.  The models (LSTM / Transformer) and training loop
(DDP, AMP, early stopping) are completely unchanged.

Usage (single GPU on Compute Canada / any Linux server):
---------------------------------------------------------
python microservice/train_sockshop.py \\
    --data_path  micro-service-trace-data/preprocessed \\
    --log_folder logs/sockshop-lstm-1 \\
    --model      lstm \\
    --train_split train_id \\
    --valid_split valid_id \\
    --ood_valid_splits "valid_ood_cpu,valid_ood_disk,valid_ood_mem" \\
    --ood_test_splits  "test_ood_cpu,test_ood_disk,test_ood_mem" \\
    --test_split test_id \\
    --n_hidden 256 --n_layer 2 \\
    --dim_sys 48 --dim_proc 48 \\
    --dim_entry 12 --dim_ret 12 \\
    --dim_pid 12 --dim_tid 12 --dim_time 12 --dim_order 12 --dim_f_mean 0 \\
    --n_categories 6 \\
    --n_update 1000000 --eval 1000 \\
    --lr 0.001 --ls 0.1 --batch 32 \\
    --gpu 0 --amp \\
    --seed 1 \\
    --train_event_model --train_latency_model \\
    --analysis
"""

import os
import sys
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

# Project root on sys.path  (microservice/ is one level down from root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microservice.NpzDataset import SockshopNpzDataset, sockshop_collate_fn
from models import LSTM, Transformer
from functions import train, evaluate, adaptive_tracing_eval

###############################################################################
# Argument parsing
###############################################################################

import argparse


def get_sockshop_args():
    p = argparse.ArgumentParser(
        description="Train LMAT on SockShop NPZ dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument("--data_path", required=True,
                   help="Path to preprocessed/ folder (contains split sub-dirs)")
    p.add_argument("--log_folder", required=True,
                   help="Directory to write logs, checkpoints, model")
    p.add_argument("--train_split",  default="train_id")
    p.add_argument("--valid_split",  default="valid_id")
    p.add_argument("--test_split",   default="test_id")
    p.add_argument("--ood_valid_splits", default="valid_ood_cpu,valid_ood_disk,valid_ood_mem",
                   help="Comma-separated list of OOD validation split names")
    p.add_argument("--ood_test_splits",  default="test_ood_cpu,test_ood_disk,test_ood_mem",
                   help="Comma-separated list of OOD test split names")

    # Model
    p.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_layer",  type=int, default=2)
    p.add_argument("--n_head",   type=int, default=4)
    p.add_argument("--n_categories", type=int, default=6)
    p.add_argument("--dropout",  type=float, default=0.01)
    p.add_argument("--dim_sys",    type=int, default=48)
    p.add_argument("--dim_proc",   type=int, default=48)
    p.add_argument("--dim_entry",  type=int, default=12)
    p.add_argument("--dim_ret",    type=int, default=12)
    p.add_argument("--dim_pid",    type=int, default=12)
    p.add_argument("--dim_tid",    type=int, default=12)
    p.add_argument("--dim_time",   type=int, default=12)
    p.add_argument("--dim_order",  type=int, default=12)
    p.add_argument("--dim_f_mean", type=int, default=0)
    p.add_argument("--activation", default="gelu", choices=["relu", "gelu", "swiglu"])
    p.add_argument("--tfixup", action="store_true")

    # Training
    p.add_argument("--gpu", type=str, default="0",
                   help="Comma-separated GPU IDs, e.g. '0' or '0,1'")
    p.add_argument("--batch",  type=int, default=32)
    p.add_argument("--n_update", type=int, default=1_000_000)
    p.add_argument("--eval",   type=int, default=1000,
                   help="Validate every N gradient updates")
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--ls",     type=float, default=0.1,
                   help="Label smoothing coefficient")
    p.add_argument("--clip",   type=float, default=10.0,
                   help="Gradient clipping max norm")
    p.add_argument("--warmup_steps",         type=int, default=0)
    p.add_argument("--reduce_lr_patience",   type=int, default=5)
    p.add_argument("--early_stopping_patience", type=int, default=20)
    p.add_argument("--amp",    action="store_true", help="AMP mixed precision")
    p.add_argument("--chk",    action="store_true", help="Gradient checkpointing")

    # Task flags
    p.add_argument("--train_event_model",   action="store_true")
    p.add_argument("--train_latency_model", action="store_true")
    p.add_argument("--ordinal_latency",     action="store_true")
    p.add_argument("--continuous_latency",  action="store_true")

    # Misc
    p.add_argument("--seed",       type=int, default=1)
    p.add_argument("--max_sample", type=int, default=None,
                   help="Max sequences to load per split (None = all)")
    p.add_argument("--max_token",  type=int, default=512)
    p.add_argument("--analysis",   action="store_true",
                   help="Run OOD detection after training")
    p.add_argument("--load_model", type=str, default=None,
                   help="Skip training and load model from this log folder")

    return p.parse_args()


###############################################################################
# Helpers
###############################################################################

def load_vocab_and_meta(data_path, train_split):
    """Load the vocabulary pickle saved by preprocess_sockshop.py."""
    vocab_path = os.path.join(data_path, "vocab.pkl")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(
            f"vocab.pkl not found in {data_path}.  "
            "Run preprocess_sockshop.py first."
        )
    with open(vocab_path, "rb") as f:
        dict_sys, dict_proc = pickle.load(f)
    return dict_sys, dict_proc


def make_dataset(data_path, split_name, max_sample, max_token, shuffle):
    split_dir = os.path.join(data_path, split_name)
    return SockshopNpzDataset(
        split_dir     = split_dir,
        max_seq_len   = max_token,
        max_samples   = max_sample,
        shuffle_shards= shuffle,
    )


###############################################################################
# Main
###############################################################################

def main():
    args = get_sockshop_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Log directory
    os.makedirs(args.log_folder, exist_ok=True)
    sys.stdout = open(os.path.join(args.log_folder, "log.txt"), "a", buffering=4096)

    # Print args
    print(f"{'='*80}\n{'SockShop LMAT Training':^80}\n{'='*80}")
    for k, v in vars(args).items():
        print(f"  {k:<30}: {v}")

    # ── Vocabulary ──────────────────────────────────────────────────────────
    dict_sys, dict_proc = load_vocab_and_meta(args.data_path, args.train_split)
    n_syscall = len(dict_sys)
    n_process = len(dict_proc)
    print(f"  Vocab: {n_syscall} syscalls, {n_process} process names")

    # ── Datasets ────────────────────────────────────────────────────────────
    train_ds = make_dataset(args.data_path, args.train_split,
                            args.max_sample, args.max_token, shuffle=True)
    valid_ds = make_dataset(args.data_path, args.valid_split,
                            args.max_sample, args.max_token, shuffle=False)
    test_ds  = make_dataset(args.data_path, args.test_split,
                            args.max_sample, args.max_token, shuffle=False)

    ood_valid_names = [s.strip() for s in args.ood_valid_splits.split(",") if s.strip()]
    ood_test_names  = [s.strip() for s in args.ood_test_splits.split(",")  if s.strip()]

    ood_valid_ds = {}
    for name in ood_valid_names:
        try:
            ood_valid_ds[name] = make_dataset(
                args.data_path, name, args.max_sample, args.max_token, False)
        except FileNotFoundError:
            print(f"  [WARN] OOD split not found: {name}")

    ood_test_ds = {}
    for name in ood_test_names:
        try:
            ood_test_ds[name] = make_dataset(
                args.data_path, name, args.max_sample, args.max_token, False)
        except FileNotFoundError:
            print(f"  [WARN] OOD test split not found: {name}")

    val_ood_to_test = {v: t for v, t in zip(ood_valid_names, ood_test_names)}

    # ── GPU setup ───────────────────────────────────────────────────────────
    gpus = list(map(int, args.gpu.split(",")))

    # ── Training ─────────────────────────────────────────────────────────────
    if args.load_model is None:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "8889"
        os.makedirs(os.path.join(args.log_folder, "training"), exist_ok=True)

        try:
            mp.spawn(
                train,
                nprocs=len(gpus),
                args=(
                    args.model,
                    n_syscall,
                    args.n_categories,
                    n_process,
                    args.n_head,
                    args.n_hidden,
                    args.n_layer,
                    args.dropout,
                    args.dim_sys,
                    args.dim_entry,
                    args.dim_ret,
                    args.dim_proc,
                    args.dim_pid,
                    args.dim_tid,
                    args.dim_order,
                    args.dim_time,
                    args.dim_f_mean,
                    args.activation,
                    args.tfixup,
                    train_ds,
                    valid_ds,
                    args.n_update,
                    args.reduce_lr_patience,
                    args.early_stopping_patience,
                    args.warmup_steps,
                    args.lr,
                    args.ls,
                    args.clip,
                    args.eval,
                    args.batch,
                    gpus,
                    args.chk,
                    args.amp,
                    args.log_folder,
                    args.train_event_model,
                    args.train_latency_model,
                    args.ordinal_latency,
                    args.continuous_latency,
                ),
            )
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            raise

    # ── Load best model ──────────────────────────────────────────────────────
    device = gpus[0]
    model_dir = args.load_model if args.load_model else args.log_folder

    if args.model == "lstm":
        model = LSTM(
            n_syscall, args.n_categories, n_process,
            args.n_hidden, args.n_layer, args.dropout,
            args.dim_sys, args.dim_entry, args.dim_ret, args.dim_proc,
            args.dim_pid, args.dim_tid, args.dim_order, args.dim_time,
            args.dim_f_mean,
            args.train_event_model, args.train_latency_model,
            args.ordinal_latency,  args.continuous_latency,
        ).to(device)
    else:
        model = Transformer(
            n_syscall, args.n_categories, n_process,
            args.n_head, args.n_hidden, args.n_layer, args.dropout,
            args.dim_sys, args.dim_entry, args.dim_ret, args.dim_proc,
            args.dim_pid, args.dim_tid, args.dim_order, args.dim_time,
            args.dim_f_mean, args.activation, args.tfixup,
            args.train_event_model, args.train_latency_model,
            args.ordinal_latency,
        ).to(device)

    model_file = os.path.join(model_dir, "model")
    with open(model_file, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=f"cuda:{device}"))
    print("Model loaded from", model_file)

    # ── Token prediction evaluation ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    if args.continuous_latency:
        criterion_latency = nn.MSELoss()
    elif args.ordinal_latency:
        criterion_latency = nn.BCEWithLogitsLoss()
    else:
        criterion_latency = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\n{'='*80}\n{'Token Prediction Evaluation':^80}\n{'='*80}")
    for name, ds in [("Train",   train_ds),
                     ("Valid ID", valid_ds),
                     ("Test ID",  test_ds)]:
        loss, acc, ll, al, ml, _ = evaluate(
            model, ds, args.batch, criterion, criterion_latency,
            n_syscall, args.n_categories, device,
            args.train_event_model, args.train_latency_model,
            args.ordinal_latency, args.continuous_latency,
        )
        print(f"  {name:<12}: loss {loss:.4f}  acc {acc:.1%}  "
              f"lat_loss {ll:.4f}  lat_acc {al:.1%}  lat_mae {ml:.4f}")

    # ── OOD detection ─────────────────────────────────────────────────────────
    if args.analysis and ood_valid_ds and ood_test_ds:
        print(f"\n{'='*80}\n{'OOD Detection':^80}\n{'='*80}")

        if args.train_event_model and args.train_latency_model:
            ev_rc, dur_rc = True, False
        elif args.train_event_model:
            ev_rc, dur_rc = True, False
        else:
            ev_rc, dur_rc = False, True

        adaptive_tracing_eval(
            model,
            (args.valid_split, valid_ds),
            {n: ood_valid_ds[n] for n in ood_valid_names if n in ood_valid_ds},
            val_ood_to_test,
            (args.test_split, test_ds),
            {n: ood_test_ds[n]  for n in ood_test_names  if n in ood_test_ds},
            args.batch,
            n_syscall,
            args.n_categories,
            device,
            args.log_folder,
            args.train_event_model,
            args.train_latency_model,
            args.ordinal_latency,
            args.continuous_latency,
            ev_rc, dur_rc,
            unique_thresh=False,
            analyze_rootCause=False,
            test_random_cases=False,
        )

    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_sockshop.py — LMAT training on SockShop NPZ dataset
==========================================================

H100-optimized training script for the SockShop anomaly detection
experiment.  Reads preprocessed NPZ shards, trains an LSTM or Transformer
model with next-syscall prediction + latency categorisation, then runs OOD
evaluation (AUROC/AUPR) on all anomaly splits.

Key features:
  - BF16 mixed-precision (H100 native)
  - torch.compile (reduce-overhead mode)
  - Gradient accumulation for large effective batch sizes
  - pin_memory + non_blocking H2D transfers
  - WandB + CSV logging
  - Cosine LR schedule with linear warm-up
  - OOD evaluation: per-anomaly-type AUROC and AUPR

Usage (single H100):
  python -u microservice/train_sockshop.py \\
      --preprocessed_dir /scratch/.../preprocessed \\
      --model transformer \\
      --n_head 8 --n_hidden 1024 --n_layer 6 \\
      --dim_sys 64 --dim_entry 8 --dim_ret 8 \\
      --dim_proc 8 --dim_pid 16 --dim_tid 16 \\
      --dim_order 16 --dim_time 16 \\
      --batch 512 --accum_steps 4 --n_epochs 20 \\
      --lr 3e-4 --warmup_steps 2000 \\
      --train_event_model --train_latency_model \\
      --amp --compile \\
      --wandb_project sockshop_lmat \\
      --log_dir logs/sockshop_exp1
"""

import os
import sys
import json
import math
import time
import random
import argparse
import csv
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch._inductor.config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# Ensure project root on sys.path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from models import LSTM, Transformer
from microservice.NpzDataset import SockshopNpzDataset, sockshop_collate_fn

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


###############################################################################
# Arguments
###############################################################################

def get_args():
    p = argparse.ArgumentParser(description="Train LMAT on SockShop NPZ dataset")

    # Data
    p.add_argument("--preprocessed_dir", required=True,
                   help="Path to preprocessed/ directory containing split subdirs")
    p.add_argument("--n_categories", type=int, default=6)
    p.add_argument("--max_seq_len",  type=int, default=512)
    p.add_argument("--max_samples",  type=int, default=None)

    # Model
    p.add_argument("--model", choices=["lstm", "transformer"], default="transformer")
    p.add_argument("--n_head",   type=int,   default=8)
    p.add_argument("--n_hidden", type=int,   default=1024)
    p.add_argument("--n_layer",  type=int,   default=6)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--activation", choices=["relu", "gelu", "swiglu"], default="gelu")
    p.add_argument("--tfixup", action="store_true")
    p.add_argument("--dim_sys",    type=int, default=64)
    p.add_argument("--dim_entry",  type=int, default=8)
    p.add_argument("--dim_ret",    type=int, default=8)
    p.add_argument("--dim_proc",   type=int, default=8)
    p.add_argument("--dim_pid",    type=int, default=16)
    p.add_argument("--dim_tid",    type=int, default=16)
    p.add_argument("--dim_order",  type=int, default=16)
    p.add_argument("--dim_time",   type=int, default=16)
    p.add_argument("--dim_f_mean", type=int, default=0)
    p.add_argument("--train_event_model",    action="store_true")
    p.add_argument("--train_latency_model",  action="store_true")
    p.add_argument("--ordinal_latency",      action="store_true")
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # Training
    p.add_argument("--batch",        type=int,   default=256)
    p.add_argument("--accum_steps",  type=int,   default=4,
                   help="Gradient accumulation steps (eff_batch = batch * accum_steps)")
    p.add_argument("--n_epochs",     type=int,   default=20)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int,   default=2000)
    p.add_argument("--clip",         type=float, default=1.0)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--amp",     action="store_true", help="BF16 mixed-precision (H100)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--chk",     action="store_true", help="Gradient checkpointing")
    p.add_argument("--seed",    type=int, default=42)

    # Logging
    p.add_argument("--log_dir",        type=str, default="logs/sockshop")
    p.add_argument("--save_every",     type=int, default=5000)
    p.add_argument("--eval_every",     type=int, default=2000)
    p.add_argument("--wandb_project",  type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--load_model",     type=str, default=None)

    # GPU — when launched via torchrun, LOCAL_RANK overrides --gpu
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index (single-GPU mode). Overridden by torchrun LOCAL_RANK.")

    args = p.parse_args()
    if not (args.train_event_model or args.train_latency_model):
        p.error("At least one of --train_event_model / --train_latency_model is required")
    return args


###############################################################################
# LR schedule — cosine with linear warm-up
###############################################################################

def lr_lambda(step, warmup_steps, total_steps, min_ratio=0.05):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


###############################################################################
# Logging
###############################################################################

class Logger:
    def __init__(self, log_dir, use_wandb, args):
        self.use_wandb = use_wandb and HAS_WANDB
        self.csv_path  = os.path.join(log_dir, "metrics.csv")
        self._writer   = None
        self._file     = None
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or
                     f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}",
                config=vars(args),
                dir=log_dir,
            )

    def log(self, metrics: dict, step: int):
        if self.use_wandb:
            wandb.log(metrics, step=step)
        if self._writer is None:
            self._file   = open(self.csv_path, "w", newline="", buffering=1)
            self._writer = csv.DictWriter(
                self._file, fieldnames=["step"] + list(metrics.keys()))
            self._writer.writeheader()
        row = {"step": step}
        row.update({k: f"{v:.6g}" if isinstance(v, float) else v
                    for k, v in metrics.items()})
        self._writer.writerow(row)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
        if self.use_wandb:
            wandb.save(path)

    def close(self):
        if self._file:
            self._file.close()
        if self.use_wandb:
            wandb.finish()


###############################################################################
# Model factory
###############################################################################

def build_model(args, n_syscall, n_process, device):
    kw = dict(
        n_syscall=n_syscall, n_category=args.n_categories, n_process=n_process,
        n_hidden=args.n_hidden, n_layer=args.n_layer, dropout=args.dropout,
        dim_sys=args.dim_sys, dim_entry=args.dim_entry, dim_ret=args.dim_ret,
        dim_proc=args.dim_proc, dim_pid=args.dim_pid, dim_tid=args.dim_tid,
        dim_order=args.dim_order, dim_time=args.dim_time, dim_f_mean=args.dim_f_mean,
        train_event=args.train_event_model, train_latency=args.train_latency_model,
        ordinal_latency=args.ordinal_latency,
    )
    if args.model == "lstm":
        return LSTM(**kw).to(device)
    return Transformer(n_head=args.n_head, activation=args.activation,
                       tfixup=args.tfixup, **kw).to(device)


###############################################################################
# Forward pass helper
###############################################################################

def forward_batch(model, batch, device, args):
    def t(key, dtype=torch.long):
        return batch[key].to(device, dtype=dtype, non_blocking=True)

    call     = t("call")
    entry    = t("entry")
    duration = t("duration")
    proc     = t("proc")
    pid      = t("pid")
    tid      = t("tid")
    ret      = t("ret")
    pad_mask = batch["pad_mask"].to(device, non_blocking=True)

    if args.model == "transformer":
        return model(call, entry, duration, proc, pid, tid, ret,
                     pad_mask=pad_mask, chk=args.chk)
    return model(call, entry, duration, proc, pid, tid, ret)


###############################################################################
# Loss
###############################################################################

def compute_loss(logits_e, logits_l, batch, device, args, crit_e, crit_l):
    tgt_call = batch["tgt_call"].to(device, dtype=torch.long, non_blocking=True)
    tgt_lat  = batch["tgt_lat" ].to(device, dtype=torch.long, non_blocking=True)
    loss = torch.tensor(0.0, device=device)
    loss_e = loss_l = torch.tensor(0.0, device=device)

    if args.train_event_model and crit_e and logits_e.numel() > 0:
        B, L, V = logits_e.shape
        loss_e = crit_e(logits_e.reshape(B*L, V), tgt_call.reshape(B*L))
        loss   = loss + loss_e

    if args.train_latency_model and crit_l and logits_l.numel() > 0:
        B, L, C = logits_l.shape
        if args.ordinal_latency:
            loss_l = crit_l(logits_l.reshape(B*L, C),
                            tgt_lat.reshape(B*L, 1).float().expand(B*L, C))
        else:
            loss_l = crit_l(logits_l.reshape(B*L, C), tgt_lat.reshape(B*L))
        loss = loss + loss_l

    return loss, loss_e, loss_l


###############################################################################
# Accuracy
###############################################################################

def token_accuracy(logits, targets):
    if logits.numel() == 0:
        return float("nan")
    pred = logits.argmax(-1)
    mask = targets != 0
    if not mask.any():
        return float("nan")
    return (pred[mask] == targets[mask]).float().mean().item()


###############################################################################
# Evaluation
###############################################################################

@torch.no_grad()
def evaluate_split(model, loader, device, args, crit_e, crit_l,
                   return_scores=False):
    model.eval()
    tot_loss = tot_e = tot_l = 0.0
    ae_sum = ae_cnt = al_sum = al_cnt = 0.0
    n = 0
    scores, labels = [], []

    for batch in loader:
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
            le, ll = forward_batch(model, batch, device, args)
            loss, loss_e, loss_l = compute_loss(le, ll, batch, device, args, crit_e, crit_l)

        tot_loss += loss.item()
        tot_e    += loss_e.item()
        tot_l    += loss_l.item()
        tgt_call  = batch["tgt_call"].to(device, dtype=torch.long, non_blocking=True)
        tgt_lat   = batch["tgt_lat" ].to(device, dtype=torch.long, non_blocking=True)
        if args.train_event_model and le.numel() > 0:
            v = token_accuracy(le, tgt_call)
            if not math.isnan(v): ae_sum += v; ae_cnt += 1
        if args.train_latency_model and ll.numel() > 0:
            v = token_accuracy(ll, tgt_lat)
            if not math.isnan(v): al_sum += v; al_cnt += 1

        if return_scores and args.train_event_model and le.numel() > 0:
            B, L, V = le.shape
            per_tok = nn.CrossEntropyLoss(ignore_index=0, reduction="none")(
                le.reshape(B*L, V), tgt_call.reshape(B*L)).reshape(B, L)
            mask = tgt_call != 0
            seq_sc = (per_tok * mask).sum(1) / mask.sum(1).clamp(min=1)
            scores.append(seq_sc.cpu().numpy())
            labels.append(batch["is_anomaly"].numpy())

        n += 1

    out = dict(loss=tot_loss/max(n,1), loss_e=tot_e/max(n,1),
               loss_l=tot_l/max(n,1),
               acc_e=ae_sum/max(ae_cnt,1), acc_l=al_sum/max(al_cnt,1))
    if return_scores:
        out["scores"] = np.concatenate(scores) if scores else np.array([])
        out["labels"] = np.concatenate(labels) if labels else np.array([])
    return out


###############################################################################
# OOD evaluation
###############################################################################

def run_ood_eval(model, args, device, crit_e, crit_l, log_fn):
    if not HAS_SKLEARN:
        log_fn("[OOD] sklearn not available — skipping")
        return {}

    base = args.preprocessed_dir
    id_dir = os.path.join(base, "test_id")
    if not os.path.isdir(id_dir):
        log_fn("[OOD] test_id not found")
        return {}

    log_fn("[OOD] Scoring test_id (normal baseline) ...")
    ds_id  = SockshopNpzDataset(id_dir, batch_size=args.batch,
                                 max_seq_len=args.max_seq_len, shuffle_shards=False)
    ld_id  = DataLoader(ds_id, batch_size=None, collate_fn=sockshop_collate_fn,
                        num_workers=2, pin_memory=True)
    res_id = evaluate_split(model, ld_id, device, args, crit_e, crit_l,
                             return_scores=True)

    results = {}
    for atype in ["cpu", "disk", "mem", "net"]:
        ood_dir = os.path.join(base, f"test_ood_{atype}")
        if not os.path.isdir(ood_dir):
            continue
        log_fn(f"[OOD] Scoring test_ood_{atype} ...")
        ds_ood = SockshopNpzDataset(ood_dir, batch_size=args.batch,
                                     max_seq_len=args.max_seq_len, shuffle_shards=False)
        ld_ood = DataLoader(ds_ood, batch_size=None, collate_fn=sockshop_collate_fn,
                            num_workers=2, pin_memory=True)
        res_ood = evaluate_split(model, ld_ood, device, args, crit_e, crit_l,
                                  return_scores=True)

        scores = np.concatenate([res_id["scores"], res_ood["scores"]])
        labels = np.concatenate([np.zeros(len(res_id["scores"])),
                                  np.ones(len(res_ood["scores"]))])
        if len(np.unique(labels)) < 2:
            continue

        auroc = roc_auc_score(labels, scores)
        aupr  = average_precision_score(labels, scores)
        log_fn(f"[OOD] {atype:6s}  AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
               f"(normal={len(res_id['scores']):,}  ood={len(res_ood['scores']):,})")
        results[atype] = {"auroc": auroc, "aupr": aupr,
                          "n_normal": len(res_id["scores"]),
                          "n_ood":    len(res_ood["scores"])}
    return results


###############################################################################
# Main
###############################################################################

def main():
    args = get_args()

    # ── DDP / single-GPU setup ─────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp_enabled = local_rank >= 0

    if ddp_enabled:
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank       = dist.get_rank()
        device     = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank       = 0
        world_size = 1
        device     = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)   # only rank 0 logs, saves, and writes WandB

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    if is_main:
        os.makedirs(args.log_dir, exist_ok=True)
    if ddp_enabled:
        dist.barrier()   # wait for rank 0 to create log_dir

    log_file = open(os.path.join(args.log_dir, f"train_rank{rank}.log"), "a", buffering=1)

    def log(msg):
        line = f"[{datetime.now():%H:%M:%S}][rank{rank}] {msg}"
        if is_main:
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()
        elif "ERROR" in msg or "WARN" in msg:
            # non-main ranks only print errors
            print(line, flush=True)

    use_wandb = bool(args.wandb_project) and HAS_WANDB and is_main
    logger    = Logger(args.log_dir, use_wandb, args) if is_main else None

    # Vocab
    with open(os.path.join(args.preprocessed_dir, "vocab.pkl"), "rb") as f:
        dict_sys, dict_proc = pickle.load(f)
    n_syscall = len(dict_sys)
    n_process = len(dict_proc)
    log(f"Vocab: {n_syscall} syscalls / {n_process} processes")
    if ddp_enabled:
        log(f"DDP: world_size={world_size}  local_rank={local_rank}  device={device}")

    # Datasets + loaders
    def make_loader(split, shuffle=True, workers=None):
        d = os.path.join(args.preprocessed_dir, split)
        if not os.path.isdir(d):
            return None, None
        ds = SockshopNpzDataset(d, batch_size=args.batch,
                                 max_seq_len=args.max_seq_len,
                                 max_samples=args.max_samples,
                                 shuffle_shards=shuffle)
        ld = DataLoader(ds, batch_size=None, collate_fn=sockshop_collate_fn,
                        num_workers=workers if workers is not None else args.num_workers,
                        pin_memory=device.type=="cuda",
                        prefetch_factor=2 if (workers or args.num_workers) > 0 else None,
                        persistent_workers=(workers or args.num_workers) > 0)
        return ds, ld

    train_ds, train_loader = make_loader("train_id",  shuffle=True)
    valid_ds, valid_loader = make_loader("valid_id",  shuffle=False)

    # Assign DDP rank so each GPU reads disjoint shards
    if ddp_enabled:
        train_ds.rank       = rank
        train_ds.world_size = world_size
        if valid_ds is not None:
            valid_ds.rank       = rank
            valid_ds.world_size = world_size

    log(f"Train shards for this rank: {len(train_ds._shards) // world_size}  "
        f"(total={len(train_ds._shards)})  Valid shards: {len(valid_ds._shards) if valid_ds else 0}")

    # Model
    model = build_model(args, n_syscall, n_process, device)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
        log(f"Loaded checkpoint from {args.load_model}")

    # Compile model
    if args.compile and torch.cuda.is_available():
        log("Compiling model with torch.compile ...")
        # Disable CUDAGraphs to prevent the "overwritten by subsequent run" embedding bug
        torch._inductor.config.triton.cudagraphs = False
        model = torch.compile(model, mode="reduce-overhead")

    # Wrap in DDP after compile
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        log(f"Model: {args.model.upper()}  params={n_params:,}")
        log(f"  n_hidden={args.n_hidden}  n_layer={args.n_layer}  n_head={args.n_head}")
        log(f"  AMP(bf16)={args.amp}  compile={args.compile}  chk={args.chk}  DDP={ddp_enabled}")
        log(f"  batch={args.batch}  accum_steps={args.accum_steps}  world_size={world_size}  "
            f"eff_batch={args.batch * args.accum_steps * world_size}")

    # Loss criteria
    crit_e = (nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
              if args.train_event_model else None)
    crit_l = ((nn.BCEWithLogitsLoss() if args.ordinal_latency
               else nn.CrossEntropyLoss(ignore_index=0))
              if args.train_latency_model else None)
    _crit_e = crit_e or nn.CrossEntropyLoss(ignore_index=0)
    _crit_l = crit_l or nn.CrossEntropyLoss(ignore_index=0)

    # Optimizer + schedule
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    steps_per_epoch = len(train_ds)
    total_steps     = steps_per_epoch * args.n_epochs // max(args.accum_steps, 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: lr_lambda(s, args.warmup_steps, total_steps))
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    with open(os.path.join(args.log_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    if ddp_enabled:
        dist.barrier()   # all ranks ready before training

    log("=" * 70)
    log(f"Starting training  device={device}  total_steps~={total_steps}  world_size={world_size}")
    log("=" * 70)

    global_step   = 0
    best_val_loss = float("inf")
    t_start       = time.time()

    for epoch in range(1, args.n_epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        acc_e_sum = acc_e_cnt = 0.0
        acc_l_sum = acc_l_cnt = 0.0
        n_batches  = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = (tqdm(train_loader, desc=f"Ep{epoch:02d}/{args.n_epochs}",
                     dynamic_ncols=True, leave=True)
                if HAS_TQDM else train_loader)

        for batch in pbar:
            is_optim_step = ((n_batches + 1) % args.accum_steps == 0)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=args.amp and device.type == "cuda"):
                logits_e, logits_l = forward_batch(model, batch, device, args)
                loss, loss_e, loss_l = compute_loss(
                    logits_e, logits_l, batch, device, args, crit_e, crit_l)
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()

            if is_optim_step:
                scaler.unscale_(optimizer)
                if args.clip:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            raw_loss = loss.item() * args.accum_steps
            epoch_loss += raw_loss
            tgt_call = batch["tgt_call"].to(device, dtype=torch.long, non_blocking=True)
            tgt_lat  = batch["tgt_lat" ].to(device, dtype=torch.long, non_blocking=True)
            if args.train_event_model and logits_e.numel() > 0:
                v = token_accuracy(logits_e.detach(), tgt_call)
                if not math.isnan(v):
                    acc_e_sum += v; acc_e_cnt += 1
            if args.train_latency_model and logits_l.numel() > 0:
                v = token_accuracy(logits_l.detach(), tgt_lat)
                if not math.isnan(v):
                    acc_l_sum += v; acc_l_cnt += 1
            n_batches += 1

            epoch_acc_e = acc_e_sum / max(acc_e_cnt, 1)
            epoch_acc_l = acc_l_sum / max(acc_l_cnt, 1)

            cur_lr = scheduler.get_last_lr()[0]
            if HAS_TQDM:
                pbar.set_postfix(
                    loss=f"{raw_loss:.3f}",
                    acc_e=f"{epoch_acc_e:.2%}" if args.train_event_model else "-",
                    lr=f"{cur_lr:.2e}",
                    step=global_step)

            # Log every 100 optimizer steps (rank 0 only)
            if is_main and is_optim_step and global_step % 100 == 0:
                metrics = {
                    "train/loss":  epoch_loss / n_batches,
                    "train/acc_e": epoch_acc_e,
                    "train/acc_l": epoch_acc_l,
                    "train/lr":    cur_lr,
                    "train/epoch": epoch,
                }
                if logger: logger.log(metrics, step=global_step)
                if global_step % 500 == 0:
                    elapsed = timedelta(seconds=int(time.time() - t_start))
                    log(f"step={global_step:6d}  epoch={epoch}  "
                        f"loss={epoch_loss/n_batches:.4f}  "
                        f"acc_e={epoch_acc_e:.2%}  "
                        f"lr={cur_lr:.2e}  elapsed={elapsed}")

            # Validation — run on all ranks, but only log from rank 0
            if (valid_loader and is_optim_step and
                    global_step > 0 and global_step % args.eval_every == 0):
                # Sync before eval so all ranks use same model state
                if ddp_enabled: dist.barrier()
                if is_main:
                    log(f"--- Validation @ step {global_step} ---")
                    res = evaluate_split(model, valid_loader, device, args, _crit_e, _crit_l)
                    log(f"  val loss={res['loss']:.4f}  "
                        f"loss_e={res['loss_e']:.4f}  acc_e={res['acc_e']:.2%}  "
                        f"acc_l={res['acc_l']:.2%}")
                    if logger: logger.log({f"val/{k}": v for k, v in res.items()
                                 if not isinstance(v, np.ndarray)},
                                step=global_step)
                    if res["loss"] < best_val_loss:
                        best_val_loss = res["loss"]
                        best_path = os.path.join(args.log_dir, "model_best.pt")
                        raw = (model.module if ddp_enabled else
                               model._orig_mod if hasattr(model, "_orig_mod") else model)
                        if logger: logger.save_model(raw, best_path)
                        log(f"  New best val loss={best_val_loss:.4f} -> {best_path}")
                if ddp_enabled: dist.barrier()
                model.train()

            # Periodic checkpoint (rank 0 only)
            if (is_main and is_optim_step and global_step > 0 and
                    global_step % args.save_every == 0):
                ckpt = os.path.join(args.log_dir, f"ckpt_{global_step:07d}.pt")
                raw  = (model.module if ddp_enabled else
                        model._orig_mod if hasattr(model, "_orig_mod") else model)
                if logger: logger.save_model(raw, ckpt)
                log(f"  Checkpoint -> {ckpt}")

        # End of epoch
        elapsed = timedelta(seconds=int(time.time() - t_start))
        log(f"Epoch {epoch:3d}/{args.n_epochs}  "
            f"loss={epoch_loss/max(n_batches,1):.4f}  "
            f"acc_e={epoch_acc_e:.2%}  "
            f"acc_l={epoch_acc_l:.2%}  "
            f"elapsed={elapsed}")
        if is_main:
            raw = (model.module if ddp_enabled else
                   model._orig_mod if hasattr(model, "_orig_mod") else model)
            if logger: logger.save_model(raw, os.path.join(args.log_dir, f"ckpt_epoch{epoch:03d}.pt"))

    # ── OOD eval + final save (rank 0 only) ─────────────────────────────────
    if is_main:
        log("=" * 70)
        log("Training complete — OOD evaluation")
        log("=" * 70)
        raw = (model.module if ddp_enabled else
               model._orig_mod if hasattr(model, "_orig_mod") else model)
        best_path = os.path.join(args.log_dir, "model_best.pt")
        if os.path.isfile(best_path):
            raw.load_state_dict(torch.load(best_path, map_location=device))
            log(f"Loaded best model from {best_path}")

        ood_results = run_ood_eval(raw, args, device, _crit_e, _crit_l, log)
        with open(os.path.join(args.log_dir, "ood_results.json"), "w") as f:
            json.dump(ood_results, f, indent=2)
        log(f"OOD results -> {os.path.join(args.log_dir, 'ood_results.json')}")

        if use_wandb and ood_results:
            wandb.log({"ood/" + at + "/" + m: v
                       for at, metrics in ood_results.items()
                       for m, v in metrics.items()})

        if logger: logger.close()
        log_file.close()

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

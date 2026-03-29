#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root on sys.path when executed as a script helper.
_HERE = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(_HERE.parent))

from microservice.NpzDataset import SockshopNpzDataset, sockshop_collate_fn
from microservice.train_sockshop import (
    _compute_mad_stats,
    _scores_to_numpy,
    build_model,
    combine_balanced_binary_scores_labels,
    combine_paper_ood_scores,
    compute_loss,
    forward_batch,
    per_sequence_event_ce,
    per_sequence_latency_ce,
)


ANOMALY_TYPES = ("cpu", "disk", "mem", "net")
MISS_LABEL = "__missed__"
UNKNOWN_LABEL = "__unknown__"


@dataclass
class RootCauseCalibration:
    threshold: float | None
    val_f1: float | None
    mad_event: dict[str, float] | None
    mad_latency: dict[str, float] | None
    n_val_normal: int
    n_val_ood: int


def load_vocab(preprocessed_dir: str) -> tuple[dict, dict]:
    vocab_path = os.path.join(preprocessed_dir, "vocab.pkl")
    with open(vocab_path, "rb") as f:
        return pickle.load(f)


def invert_vocab(vocab: dict) -> dict[int, str]:
    if hasattr(vocab, "word2idx"):
        items = vocab.word2idx.items()
    elif isinstance(vocab, dict):
        items = vocab.items()
    else:
        raise TypeError(f"Unsupported vocab type for inversion: {type(vocab)!r}")

    out = {}
    for key, value in items:
        out[int(value)] = str(key)
    return out


def make_loader(
    preprocessed_dir: str,
    subdir: str,
    batch: int,
    max_seq_len: int,
    max_samples: int | None = None,
    num_workers: int = 2,
    pin_memory: bool | None = None,
):
    split_dir = os.path.join(preprocessed_dir, subdir)
    if not os.path.isdir(split_dir):
        return None
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    ds = SockshopNpzDataset(
        split_dir,
        batch_size=batch,
        max_seq_len=max_seq_len,
        max_samples=max_samples,
        shuffle_shards=False,
    )
    return DataLoader(
        ds,
        batch_size=None,
        collate_fn=sockshop_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_model_from_args(args, device: torch.device):
    dict_sys, dict_proc = load_vocab(args.preprocessed_dir)
    model = build_model(args, len(dict_sys), len(dict_proc), device)
    state = torch.load(args.load_model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    id_to_syscall = invert_vocab(dict_sys)
    return model, dict_sys, dict_proc, id_to_syscall


def decode_latency_predictions(logits_l: torch.Tensor, ordinal_latency: bool) -> torch.Tensor | None:
    if logits_l.numel() == 0:
        return None
    if ordinal_latency:
        probs = torch.sigmoid(logits_l)
        return (probs > 0.5).sum(-1).to(dtype=torch.long) + 1
    return logits_l.argmax(-1)


def count_event_errors(
    pred_call: torch.Tensor | None,
    tgt_call: torch.Tensor,
    vocab_size: int,
) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    if pred_call is None:
        return vec
    mask = tgt_call != 0
    if not torch.any(mask):
        return vec
    mismatch = mask & (pred_call != tgt_call)
    if not torch.any(mismatch):
        return vec
    event_ids = tgt_call[mismatch].detach().to(dtype=torch.long).cpu().numpy()
    counts = np.bincount(event_ids, minlength=vocab_size)
    vec[: len(counts)] = counts.astype(np.float32, copy=False)
    vec[0] = 0.0
    return vec


def count_latency_errors(
    pred_lat: torch.Tensor | None,
    tgt_lat: torch.Tensor,
    tgt_call: torch.Tensor,
    vocab_size: int,
) -> np.ndarray:
    vec = np.zeros(vocab_size, dtype=np.float32)
    if pred_lat is None:
        return vec
    mask = (tgt_lat != 0) & (tgt_call != 0)
    if not torch.any(mask):
        return vec
    mismatch = mask & (pred_lat != tgt_lat)
    if not torch.any(mismatch):
        return vec
    event_ids = tgt_call[mismatch].detach().to(dtype=torch.long).cpu().numpy()
    counts = np.bincount(event_ids, minlength=vocab_size)
    vec[: len(counts)] = counts.astype(np.float32, copy=False)
    vec[0] = 0.0
    return vec


def combine_error_vectors(
    event_vec: np.ndarray,
    latency_vec: np.ndarray,
    has_event: bool,
    has_latency: bool,
    strategy: str = "mean",
) -> np.ndarray:
    event_arr = np.asarray(event_vec, dtype=np.float32)
    latency_arr = np.asarray(latency_vec, dtype=np.float32)
    if has_event and has_latency:
        if strategy == "concat":
            return np.concatenate([event_arr, latency_arr]).astype(np.float32, copy=False)
        if strategy == "sum":
            return (event_arr + latency_arr).astype(np.float32, copy=False)
        return ((event_arr + latency_arr) / 2.0).astype(np.float32, copy=False)
    if has_event:
        return event_arr.astype(np.float32, copy=False)
    if has_latency:
        return latency_arr.astype(np.float32, copy=False)
    return np.array([], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


@torch.no_grad()
def extract_root_cause_records(
    model,
    loader,
    device: torch.device,
    args,
    split_name: str,
    anomaly_label: str | None,
    id_to_syscall: dict[int, str],
    score_threshold: float | None = None,
    mad_event: dict[str, float] | None = None,
    mad_latency: dict[str, float] | None = None,
    combine_strategy: str = "mean",
):
    records = []
    seq_index = 0
    vocab_size = max(id_to_syscall.keys()) + 1 if id_to_syscall else 1
    print(
        f"[RCA] [{split_name}] start extraction"
        + (f" ({anomaly_label})" if anomaly_label else ""),
        flush=True,
    )

    model.eval()
    for batch_idx, batch in enumerate(loader):
        if batch_idx % 100 == 0:
            print(
                f"[RCA] [{split_name}] batch={batch_idx} sequences={seq_index}",
                flush=True,
            )
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.amp and device.type == "cuda",
        ):
            logits_e, logits_l = forward_batch(model, batch, device, args)

        tgt_call = batch["tgt_call"].to(device, dtype=torch.long, non_blocking=True)
        tgt_lat = batch["tgt_lat"].to(device, dtype=torch.long, non_blocking=True)
        seq_event_scores = per_sequence_event_ce(logits_e, tgt_call) if logits_e.numel() > 0 else None
        seq_latency_scores = (
            per_sequence_latency_ce(logits_l, tgt_lat, args.ordinal_latency)
            if logits_l.numel() > 0
            else None
        )
        scores_event = _scores_to_numpy(seq_event_scores)
        scores_latency = _scores_to_numpy(seq_latency_scores)
        anomaly_scores = combine_paper_ood_scores(
            scores_event,
            scores_latency,
            args,
            mad_event,
            mad_latency,
        )

        pred_call = logits_e.argmax(-1) if logits_e.numel() > 0 else None
        pred_lat = decode_latency_predictions(logits_l, args.ordinal_latency)

        seq_len_arr = batch["seq_len"].detach().cpu().numpy()
        anomaly_arr = batch["is_anomaly"].detach().cpu().numpy()
        batch_size = int(seq_len_arr.shape[0])

        for row in range(batch_size):
            event_vec = count_event_errors(
                pred_call[row] if pred_call is not None else None,
                tgt_call[row],
                vocab_size,
            )
            latency_vec = count_latency_errors(
                pred_lat[row] if pred_lat is not None else None,
                tgt_lat[row],
                tgt_call[row],
                vocab_size,
            )
            combined_vec = combine_error_vectors(
                event_vec,
                latency_vec,
                has_event=args.train_event_model,
                has_latency=args.train_latency_model,
                strategy=combine_strategy,
            )

            event_top_id = int(np.argmax(event_vec)) if np.any(event_vec[1:] > 0) else 0
            latency_top_id = int(np.argmax(latency_vec)) if np.any(latency_vec[1:] > 0) else 0
            score = float(anomaly_scores[row]) if row < anomaly_scores.size else float("nan")
            detected = bool(score_threshold is not None and np.isfinite(score) and score > score_threshold)
            event_score = float(scores_event[row]) if row < scores_event.size else float("nan")
            latency_score = float(scores_latency[row]) if row < scores_latency.size else float("nan")

            records.append(
                {
                    "split": split_name,
                    "anomaly_label": anomaly_label or "normal",
                    "sequence_index": seq_index,
                    "batch_index": batch_idx,
                    "row_in_batch": row,
                    "seq_len": int(seq_len_arr[row]),
                    "is_anomaly": int(anomaly_arr[row]),
                    "score": score,
                    "score_event": event_score,
                    "score_latency": latency_score,
                    "detected": detected,
                    "event_error_vector": event_vec,
                    "latency_error_vector": latency_vec,
                    "combined_error_vector": combined_vec,
                    "event_error_sum": float(event_vec.sum()),
                    "latency_error_sum": float(latency_vec.sum()),
                    "combined_error_sum": float(combined_vec.sum()) if combined_vec.size > 0 else 0.0,
                    "event_top_error_id": event_top_id,
                    "event_top_error_name": id_to_syscall.get(event_top_id, str(event_top_id)),
                    "latency_top_error_id": latency_top_id,
                    "latency_top_error_name": id_to_syscall.get(latency_top_id, str(latency_top_id)),
                }
            )
            seq_index += 1

    print(
        f"[RCA] [{split_name}] finished extraction total_sequences={len(records)}",
        flush=True,
    )
    return records


def select_vector(record: dict) -> np.ndarray:
    return np.asarray(record["combined_error_vector"], dtype=np.float32)


def build_centroids(
    records_by_label: dict[str, list[dict]],
    centroid_source: str = "all",
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    centroids = {}
    metadata = {}
    for label, records in records_by_label.items():
        selected = records
        if centroid_source == "detected":
            detected_records = [r for r in records if r["detected"]]
            if detected_records:
                selected = detected_records

        vectors = [select_vector(r) for r in selected]
        vectors = [v for v in vectors if v.size > 0]
        nonzero_vectors = [v for v in vectors if float(np.linalg.norm(v)) > 1e-12]
        use_vectors = nonzero_vectors or vectors
        if not use_vectors:
            continue
        centroid = np.mean(np.stack(use_vectors, axis=0), axis=0).astype(np.float32, copy=False)
        centroids[label] = centroid
        metadata[label] = {
            "n_records_total": len(records),
            "n_records_selected": len(selected),
            "n_vectors_used": len(use_vectors),
            "n_nonzero_vectors_used": len(nonzero_vectors),
        }
    return centroids, metadata


def classify_record(record: dict, centroids: dict[str, np.ndarray]) -> tuple[str | None, float]:
    vec = select_vector(record)
    if vec.size == 0:
        return None, float("nan")
    best_label = None
    best_score = float("-inf")
    for label, centroid in centroids.items():
        sim = cosine_similarity(vec, centroid)
        if math.isnan(sim):
            continue
        if sim > best_score:
            best_label = label
            best_score = sim
    if best_label is None:
        return None, float("nan")
    return best_label, best_score


def compute_accuracy(records: list[dict], predicate) -> float | None:
    if not records:
        return None
    correct = sum(1 for r in records if predicate(r))
    return correct / len(records)


def confusion_matrix_dict(
    true_labels: list[str],
    pred_labels: list[str],
    labels: list[str],
) -> dict[str, dict[str, int]]:
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label not in matrix:
            matrix[true_label] = {p: 0 for p in labels}
        if pred_label not in matrix[true_label]:
            for row in matrix.values():
                row.setdefault(pred_label, 0)
        matrix[true_label][pred_label] += 1
    return matrix


def summarise_centroid(
    centroid: np.ndarray,
    id_to_syscall: dict[int, str],
    top_k: int = 10,
) -> list[dict]:
    arr = np.asarray(centroid, dtype=np.float64)
    if arr.size <= 1:
        return []
    idx = np.argsort(arr[1:])[::-1] + 1
    out = []
    for token_id in idx[:top_k]:
        score = float(arr[token_id])
        if score <= 0:
            break
        out.append(
            {
                "syscall_id": int(token_id),
                "syscall": id_to_syscall.get(int(token_id), str(int(token_id))),
                "weight": score,
            }
        )
    return out


def write_predictions_csv(path: str, records: list[dict]):
    fieldnames = [
        "split",
        "anomaly_label",
        "sequence_index",
        "batch_index",
        "row_in_batch",
        "seq_len",
        "is_anomaly",
        "score",
        "score_event",
        "score_latency",
        "detected",
        "predicted_label",
        "predicted_similarity",
        "detected_only_correct",
        "end_to_end_correct",
        "isolated_correct",
        "event_error_sum",
        "latency_error_sum",
        "combined_error_sum",
        "event_top_error_name",
        "latency_top_error_name",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record["split"],
                    "anomaly_label": record["anomaly_label"],
                    "sequence_index": record["sequence_index"],
                    "batch_index": record["batch_index"],
                    "row_in_batch": record["row_in_batch"],
                    "seq_len": record["seq_len"],
                    "is_anomaly": record["is_anomaly"],
                    "score": record["score"],
                    "score_event": record["score_event"],
                    "score_latency": record["score_latency"],
                    "detected": record["detected"],
                    "predicted_label": record.get("predicted_label"),
                    "predicted_similarity": record.get("predicted_similarity"),
                    "detected_only_correct": record.get("detected_only_correct"),
                    "end_to_end_correct": record.get("end_to_end_correct"),
                    "isolated_correct": record.get("isolated_correct"),
                    "event_error_sum": record["event_error_sum"],
                    "latency_error_sum": record["latency_error_sum"],
                    "combined_error_sum": record["combined_error_sum"],
                    "event_top_error_name": record["event_top_error_name"],
                    "latency_top_error_name": record["latency_top_error_name"],
                }
            )

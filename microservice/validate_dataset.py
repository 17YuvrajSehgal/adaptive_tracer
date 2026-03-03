#!/usr/bin/env python3
"""Quick validation of all preprocessed NPZ splits."""
import numpy as np
import glob, os, json, sys

BASE = os.path.join("micro-service-trace-data", "preprocessed")

SPLITS = [
    ("train_id",       0),
    ("valid_id",       0),
    ("test_id",        0),
    ("valid_ood_cpu",  1),
    ("test_ood_cpu",   1),
    ("valid_ood_disk", 1),
    ("test_ood_disk",  1),
    ("valid_ood_mem",  1),
    ("test_ood_mem",   1),
    ("valid_ood_net",  1),
    ("test_ood_net",   1),
]

REQUIRED_KEYS = [
    "call", "entry", "duration", "proc", "pid", "tid",
    "ret", "lat_cat", "seq_len", "req_dur_ms", "is_anomaly",
]

print("=" * 72)
print(f"  {'split':<18} {'shards':>6}  {'total_seqs':>9}  {'avg_sl':>6}  {'is_anom':>7}  status")
print("  " + "-" * 68)

all_ok = True
results = {}

for split, expected_label in SPLITS:
    d = os.path.join(BASE, split)
    shards = sorted(glob.glob(os.path.join(d, "shard_*.npz")))
    if not shards:
        print(f"  {split:<18} NO SHARDS FOUND")
        all_ok = False
        continue

    total, sum_sl = 0, 0
    max_l = 0
    errs = []

    for p in shards:
        s = np.load(p, allow_pickle=False)

        # Check all required keys present
        for key in REQUIRED_KEYS:
            if key not in s.files:
                errs.append(f"missing:{key}")

        n = s["seq_len"].shape[0]
        total += n
        sum_sl += int(s["seq_len"].sum())
        max_l = max(max_l, s["call"].shape[1])

        # Check label
        anom = int(s["is_anomaly"][0])
        if anom != expected_label:
            errs.append(f"is_anomaly={anom} want {expected_label}")

        # Check no sequence longer than padded dim
        if int(s["seq_len"].max()) > s["call"].shape[1]:
            errs.append("seq_len overflow")

    errs = list(dict.fromkeys(errs))  # deduplicate
    avg_sl = sum_sl / total if total else 0
    status = "OK" if not errs else "WARN: " + "; ".join(errs)
    print(f"  {split:<18} {len(shards):>6}  {total:>9,}  {avg_sl:>6.1f}  {expected_label:>7}  {status}")
    if errs:
        all_ok = False
    results[split] = {"shards": len(shards), "seqs": total, "avg_sl": avg_sl}

# ── Array shape / dtype check on train_id shard_0 ─────────────────
print()
print("=" * 72)
print("train_id / shard_000000.npz  — array details")
print("=" * 72)
s0_path = os.path.join(BASE, "train_id", "shard_000000.npz")
s0 = np.load(s0_path, allow_pickle=False)
for k in sorted(s0.files):
    print(f"  {k:<14}  dtype={str(s0[k].dtype):<8}  shape={s0[k].shape}")

sl = s0["seq_len"]
print(f"\n  seq_len  min={sl.min()}  max={sl.max()}  mean={sl.mean():.1f}  median={float(np.median(sl)):.1f}")

lc = s0["lat_cat"].flatten()
vals, cnts = np.unique(lc, return_counts=True)
dist = {int(v): f"{c/len(lc)*100:.1f}%" for v, c in zip(vals, cnts)}
print(f"  lat_cat  distribution: {dist}")

# ── Sample 3 sequences to eyeball call token IDs ─────────────────
print()
print("  First 3 sequence call token IDs (should start with 2=START):")
for i in range(3):
    slen = int(sl[i])
    tokens = s0["call"][i, :min(slen, 15)].tolist()
    print(f"    seq[{i}] len={slen:3d}: {tokens}")

# ── Anomaly label sanity: normal splits must be 0, ood must be 1 ─
print()
print("=" * 72)
print("VERDICT")
print("=" * 72)
total_seqs = sum(v["seqs"] for v in results.values())
print(f"  Total sequences across all splits : {total_seqs:,}")
print(f"  All keys present                  : {'YES' if all_ok else 'NO - see WARN above'}")
print(f"  Ready for training                : {'YES' if all_ok else 'NO'}")

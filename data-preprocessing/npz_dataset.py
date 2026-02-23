"""Drop-in replacement for ``dataset.IterableDataset`` that reads NPZ shards.

Yields the **exact same tuple** as the original text-based ``IterableDataset``::

    (call[:-1], entry[:-1], duration[:-1], proc[:-1],
     pid[:-1], tid[:-1], ret[:-1], latency[:-1],  # model inputs  (n-1)
     call[1:],                                      # next-event target (n-1)
     timestamp_ns, req_duration_ms)                 # metadata

This makes it a drop-in replacement: ``collate_fn``, the model, and the
training loop all stay unchanged.

NPZ shard format (written by ``preprocess.py``)
-----------------------------------------------
call, entry, duration, proc, pid, tid, ret :  int arrays  (N, L)
lat4, lat6, lat8, lat10                    :  int arrays  (N, L)
lengths                                    :  int array   (N,)
timestamps                                 :  int array   (N,)
req_dur                                    :  float array (N,)

Where N = requests per shard, L = max sequence length within the shard.

The ``latency`` sequence is selected from lat4/lat6/lat8/lat10 based on
``n_categories``, mirroring the original ``category_idx`` logic.

Truncation (``max_token``)
--------------------------
If a sequence is longer than ``max_token``, it is truncated and a
``[TRUNCATE]`` token (id=4) is appended to the syscall and process
sequences, while a ``[MASK]`` token (id=0) is appended to all others —
identical to the original implementation.

Distributed / multi-worker support
-----------------------------------
When used with ``DataLoader(num_workers=K)``, each worker reads a disjoint
stripe of *shard files* (not lines within a shard) which is safe because
each shard file is fully independent.
"""

from __future__ import annotations

import glob
import os
from itertools import islice
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils import data


# map n_categories → NPZ field name (matches what preprocess.py writes)
_CAT_FIELD = {4: "lat4", 6: "lat6", 8: "lat8", 10: "lat10"}


class NPZIterableDataset(data.IterableDataset):
    """Iterable dataset backed by NPZ shards.

    Args:
        shard_dir:         Path to the ``npz/`` directory produced by
                           ``preprocess.py`` for one split.
        n_categories:      Number of latency categories (must be 4, 6, 8, or
                           10).  Selects the matching latency array from the
                           shard.
        max_token:         Truncate sequences longer than this (``None`` =
                           no truncation).
        max_sample:        Maximum number of requests to read from the shard
                           directory in total (``None`` = read all).
        continuous_latency:If ``True``, return the raw duration column instead
                           of a latency-category column.  The raw durations are
                           stored in the ``duration`` field of the shard.
    """

    def __init__(
        self,
        shard_dir:          str,
        n_categories:       int  = 6,
        max_token:          Optional[int] = None,
        max_sample:         Optional[int] = None,
        continuous_latency: bool = False,
    ) -> None:
        if not continuous_latency and n_categories not in _CAT_FIELD:
            raise ValueError(
                f"n_categories must be one of {sorted(_CAT_FIELD)}, "
                f"got {n_categories}"
            )

        self.shard_dir          = shard_dir
        self.n_categories       = n_categories
        self.max_token          = max_token
        self.max_sample         = max_sample
        self.continuous_latency = continuous_latency
        self.lat_field          = _CAT_FIELD.get(n_categories, "lat6")

        # Locate all shard files once at construction time
        self._shards: List[str] = sorted(
            glob.glob(os.path.join(shard_dir, "shard_*.npz"))
        )
        if not self._shards:
            raise FileNotFoundError(
                f"No shard_*.npz files found in {shard_dir!r}. "
                "Run preprocess.py first."
            )

        # Distributed-training placeholders (set by the training harness)
        self.rank:       int = 0
        self.world_size: int = 1

    # ------------------------------------------------------------------
    # Core iteration
    # ------------------------------------------------------------------

    def _iter_shards(self) -> Tuple[List[int], ...]:
        """Yield individual requests from the shard files assigned to this
        worker / rank, respecting ``max_sample``.
        """
        worker_info = data.get_worker_info()
        if worker_info is None:
            # Single-process DataLoader
            worker_id   = self.rank
            world_total = self.world_size
        else:
            # Multi-worker DataLoader: stripe across workers × DDP ranks
            worker_id   = self.rank * worker_info.num_workers + worker_info.id
            world_total = self.world_size * worker_info.num_workers

        # Each worker reads a disjoint subset of shard files
        my_shards = self._shards[worker_id::world_total]

        served = 0
        for shard_path in my_shards:
            if self.max_sample is not None and served >= self.max_sample:
                break

            # Memory-map the shard for zero-copy reads
            shard = np.load(shard_path, allow_pickle=False, mmap_mode="r")
            lengths    = shard["lengths"]       # (N,)
            timestamps = shard["timestamps"]    # (N,)
            req_durs   = shard["req_dur"]       # (N,)

            call_s     = shard["call"]          # (N, L)
            entry_s    = shard["entry"]
            dur_s      = shard["duration"]
            proc_s     = shard["proc"]
            pid_s      = shard["pid"]
            tid_s      = shard["tid"]
            ret_s      = shard["ret"]
            lat_s      = shard[self.lat_field] if not self.continuous_latency \
                         else shard["duration"]  # fallback for continuous mode

            for i in range(len(lengths)):
                if self.max_sample is not None and served >= self.max_sample:
                    break

                L = int(lengths[i])
                # Slice to true length (avoids feeding padding to model)
                call  = call_s[i, :L].tolist()
                entry = entry_s[i, :L].tolist()
                dur   = dur_s[i, :L].tolist()
                proc  = proc_s[i, :L].tolist()
                pid   = pid_s[i, :L].tolist()
                tid   = tid_s[i, :L].tolist()
                ret   = ret_s[i, :L].tolist()
                lat   = lat_s[i, :L].tolist()

                ts    = int(timestamps[i])
                rdur  = float(req_durs[i])

                # ---- truncation (identical to original) ----------------
                if self.max_token is not None and L > self.max_token:
                    call  = call[:self.max_token];   call.append(4)   # [TRUNCATE]
                    proc  = proc[:self.max_token];   proc.append(4)
                    entry = entry[:self.max_token];  entry.append(0)  # [MASK]
                    dur   = dur[:self.max_token];    dur.append(0)
                    pid   = pid[:self.max_token];    pid.append(0)
                    tid   = tid[:self.max_token];    tid.append(0)
                    ret   = ret[:self.max_token];    ret.append(0)
                    lat   = lat[:self.max_token];    lat.append(0)

                # Assemble the "sample" list (same order as original parser)
                # sample = [call, entry, dur, proc, pid, tid, ret, lat]
                # yield: (*[x[:-1] for x in sample], call[1:], ts, rdur)
                sample = [call, entry, dur, proc, pid, tid, ret, lat]

                yield (
                    *(x[:-1] for x in sample),   # inputs  (length n-1)
                    call[1:],                     # next-event targets
                    ts,
                    rdur,
                )
                served += 1

    def __iter__(self):
        return self._iter_shards()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Approximate total request count (sum of shard sizes)."""
        total = 0
        for sp in self._shards:
            d = np.load(sp, allow_pickle=False)
            total += len(d["lengths"])
        return total if self.max_sample is None else min(total, self.max_sample)

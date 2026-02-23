"""Vectorised latency span building and categorisation.

The original LMAT code computes per-event-type latency percentile bins and
then categorises each event using a Python ``for`` loop plus an inner linear
search (``categorize_value``).

This module provides faster replacements:

* ``build_spans_for_n_cat``  – compute percentile boundaries for one n_cat.
* ``build_all_spans``        – build spans for all four n_cat sizes (4,6,8,10)
                               from an event history dict.
* ``merge_span_dicts``       – weighted-average merge of two span dicts
                               (used when updating incrementally during training).
* ``categorize_batch``       – vectorised categorisation using
                               ``np.searchsorted`` instead of a linear Python
                               loop; returns categories for all four n_cat
                               sizes in one pass.

The percentile formula is identical to the original::

    n_bins    = n_categories - 1          # usable bins (category 0 = padding)
    total     = n_bins * (n_bins + 1) / 2
    for i in range(1, n_bins):
        fraction     = (n_bins - i + 1) / total
        cumulative  += fraction * 100
        percentiles.append(cumulative)
    boundaries = np.percentile(sorted_delays, percentiles)

The categorisation rule (also identical to the original)::

    category = np.searchsorted(boundaries, value, side='right') + 1
    # i.e. the interval [boundary[k-1], boundary[k]] → category k+1

Special cases that match the original exactly:
    * Entry events                    → category 0 (no duration)
    * Null latency (0)                → category 0 (no paired entry found)
    * Unknown event name in spans     → category 0
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple

# The four category sizes LMAT stores in parallel (matches the original loop
#   ``for i in range(4): n_cat = i * 2 + 4``).
ALL_N_CATS: Tuple[int, ...] = (4, 6, 8, 10)


# ---------------------------------------------------------------------------
# Building latency span boundaries
# ---------------------------------------------------------------------------

def build_spans_for_n_cat(
    event_time_map_hist: Dict[str, List[int]],
    n_categories: int,
) -> Dict[str, Tuple[np.ndarray, int]]:
    """Compute percentile-based latency boundaries for each event type.

    Replicates the ``update_spans=True`` branch of the original
    ``categorize_latency`` function exactly.

    Args:
        event_time_map_hist: Mapping from (normalised) event name to a list
            of observed exit-entry durations in nanoseconds.
        n_categories: Total category count *including* the padding category 0.
            Effective bins = n_categories - 1.  Must be one of 4, 6, 8, 10.

    Returns:
        Dict mapping event name → (boundaries_array, observation_count).
        ``boundaries_array`` has shape ``(n_bins - 1,)`` and is sorted.
    """
    n_bins = n_categories - 1                     # e.g. 5 for n_categories=6
    total  = n_bins * (n_bins + 1) / 2

    # Build the list of percentile cut-points (matches the original loop)
    pct_points: List[float] = []
    cumulative = 0.0
    for i in range(1, n_bins):
        fraction    = (n_bins - i + 1) / total
        cumulative += fraction * 100.0
        pct_points.append(cumulative)

    spans: Dict[str, Tuple[np.ndarray, int]] = {}
    for event_name, delays in event_time_map_hist.items():
        if not delays:
            continue
        arr        = np.sort(np.array(delays, dtype=np.float64))
        boundaries = np.percentile(arr, pct_points)
        spans[event_name] = (boundaries, len(arr))

    return spans


def build_all_spans(
    event_time_map_hist: Dict[str, List[int]],
) -> List[Optional[Dict[str, Tuple[np.ndarray, int]]]]:
    """Build spans for all four n_cat sizes (4, 6, 8, 10) in one call.

    Returns:
        List of four span dicts, one per n_cat in ``ALL_N_CATS`` order.
        Returns ``[None, None, None, None]`` if ``event_time_map_hist`` is
        empty (train set not yet started).
    """
    if not event_time_map_hist:
        return [None, None, None, None]
    return [build_spans_for_n_cat(event_time_map_hist, nc) for nc in ALL_N_CATS]


def merge_span_dicts(
    existing: Optional[Dict[str, Tuple[np.ndarray, int]]],
    new_spans: Dict[str, Tuple[np.ndarray, int]],
) -> Dict[str, Tuple[np.ndarray, int]]:
    """Merge *new_spans* into *existing* using a weighted average.

    Replicates the incremental update logic from the original
    ``categorize_latency`` (the ``else`` branch after ``tmp_delay_spans``
    is built):  each event's boundary array is updated as a weighted average
    of the old and new arrays, weighted by their respective observation counts.

    Args:
        existing:  Previously accumulated spans dict (or ``None`` for first
                   batch).
        new_spans: Spans computed from the current batch.

    Returns:
        Merged spans dict.
    """
    if existing is None:
        return dict(new_spans)

    merged = dict(existing)
    for name, (new_bounds, new_cnt) in new_spans.items():
        if name in merged:
            prev_bounds, prev_cnt = merged[name]
            total   = prev_cnt + new_cnt
            w_bounds = (prev_bounds * prev_cnt + new_bounds * new_cnt) / total
            merged[name] = (w_bounds, total)
        else:
            merged[name] = (new_bounds, new_cnt)
    return merged


def merge_all_spans(
    existing_list: List[Optional[Dict]],
    new_hist: Dict[str, List[int]],
) -> List[Dict]:
    """Update all four span dicts with data from *new_hist*.

    Args:
        existing_list: Four-element list from a previous call (or four Nones).
        new_hist:      Current ``event_time_map_hist`` snapshot.

    Returns:
        Updated four-element list.
    """
    new_list = build_all_spans(new_hist)
    return [
        merge_span_dicts(ex, nw) if nw else ex
        for ex, nw in zip(existing_list, new_list)
    ]


# ---------------------------------------------------------------------------
# Fast vectorised categorisation
# ---------------------------------------------------------------------------

def categorize_batch(
    latency_arr:       List[List[int]],
    event_names_arr:   List[List[str]],
    span_list:         List[Optional[Dict[str, Tuple[np.ndarray, int]]]],
) -> Dict[int, List[List[int]]]:
    """Categorise a batch of requests for all four n_cat sizes.

    Key speedup over the original: instead of a Python ``for`` loop over
    boundaries (``categorize_value``), we group events by normalised name,
    stack their latencies into a NumPy array, and call ``np.searchsorted``
    once per (event_type, n_cat) pair.  For typical requests with many
    occurrences of the same syscall this reduces the number of Python-level
    iterations by O(events_per_type).

    The categorisation formula is:
        category = np.searchsorted(boundaries, latency, side='right') + 1
    which is identical to the original linear search because ``boundaries``
    is sorted (see module docstring for proof).

    Special cases (all producing category 0):
        * Event name not in ``span_list`` → unknown → 0
        * Event is an entry event (not "exit" in name) → 0
        * ``latency == 0`` for an exit event (missing entry) → kept as-is
          and categorised normally (the original also categorises these).

    Args:
        latency_arr:     Outer list = requests; inner list = per-event raw
                         latency in ns.
        event_names_arr: Matching list of full event names (e.g.
                         ``"syscall_exit_read"``).
        span_list:       Four-element list of span dicts, one per n_cat in
                         ``ALL_N_CATS`` order.

    Returns:
        Dict ``{4: [[...], ...], 6: [...], 8: [...], 10: [...]}``.
        Each value is a list of per-request category lists (same shape as
        *latency_arr*).
    """
    results: Dict[int, List[List[int]]] = {nc: [] for nc in ALL_N_CATS}

    for req_latencies, req_event_names in zip(latency_arr, event_names_arr):
        n_events = len(req_latencies)

        # Initialise per-n_cat category arrays for this request
        req_cats: Dict[int, np.ndarray] = {
            nc: np.zeros(n_events, dtype=np.int32) for nc in ALL_N_CATS
        }

        # --------------- group indices by normalised event name -------------
        # Only exit events get a non-zero category (entry → 0).
        # Group them to enable batch np.searchsorted.
        from collections import defaultdict
        groups: Dict[str, List[int]] = defaultdict(list)  # norm_name → [idx]
        exit_latencies: Dict[int, int] = {}               # idx → latency_ns

        for idx, (lat, ev_name) in enumerate(zip(req_latencies, req_event_names)):
            if "exit" not in ev_name:
                continue  # entry / neutral → stays 0
            norm = (
                ev_name
                .replace("syscall_", "")
                .replace("entry_",   "")
                .replace("exit_",    "")
            )
            groups[norm].append(idx)
            exit_latencies[idx] = lat

        # --------------- vectorised categorisation per event type -----------
        for norm_name, indices in groups.items():
            arr = np.array(
                [exit_latencies[i] for i in indices], dtype=np.float64
            )
            for nc, spans in zip(ALL_N_CATS, span_list):
                if spans is None:
                    continue
                span_info = spans.get(norm_name)
                if span_info is None:
                    continue  # unknown event → 0
                boundaries = span_info[0]            # sorted np.ndarray
                cats = np.searchsorted(boundaries, arr, side="right") + 1
                req_cats[nc][indices] = cats.astype(np.int32)

        for nc in ALL_N_CATS:
            results[nc].append(req_cats[nc].tolist())

    return results

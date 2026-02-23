"""Time-window-based trace reader for kernel-only traces (e.g. Train-Ticket).

The Apache LMAT pipeline uses ``httpd:enter/exit_event_handler`` events to
delineate HTTP requests.  Microservice kernel traces have no such markers.
This module replaces ``get_requests`` with **per-PID fixed time windows**:

    sequence = all syscalls made by one PID during one time window

This preserves the same structural pattern LMAT learned: a sequence of
entry/exit syscall pairs, with timing and return-value metadata.  Anomalies
(CPU stress, IO flood, memory pressure) alter the distribution of syscall
types and durations within a window, exactly as LMAT expects.

Segmentation rules
------------------
* Global time is sliced into non-overlapping windows of ``window_ns`` ns.
* When the trace crosses a window boundary, every PID whose buffer contains
  >= ``min_events`` events is yielded as a completed sequence and flushed.
* PIDs with < ``min_events`` events in a window are discarded (idle threads).
* ``event_time_map`` (for entry→exit duration pairing) is *never* reset across
  window boundaries — an entry in window N may pair with an exit in window N+1.

Public API
----------
    get_events_tt(trace, keys)          → same as trace_reader.get_events
    get_sequences_by_pid_window(...)    → generator of (event_list, procname,
                                          pid, window_start_ns)
"""

from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# Re-export get_events unchanged — the event format is identical
from trace_reader import get_events, get_duration, load_trace  # noqa: F401


# ---------------------------------------------------------------------------
# Per-PID time-window segmenter
# ---------------------------------------------------------------------------

def get_sequences_by_pid_window(
    events: Generator[Dict[str, Any], None, None],
    window_ns:   int = 500_000_000,   # 500 ms
    min_events:  int = 10,
    filter_procs: Optional[Set[str]] = None,
) -> Generator[Tuple[List[Dict], str, int, int], None, None]:
    """Segment a kernel-event stream into per-PID time windows.

    Args:
        events:       Generator from ``get_events`` (global timestamp order).
        window_ns:    Window width in nanoseconds.  Default 500 ms.
        min_events:   Windows with fewer events are dropped (idle threads).
        filter_procs: Optional set of ``procname`` substrings.  A PID is
                      included only if its last-seen procname contains any
                      string in this set.  ``None`` = include all PIDs.

    Yields:
        (event_list, procname, pid, window_start_ns)
        event_list      – ordered list of event dicts for one (PID, window).
        procname        – most recently seen procname for this PID.
        pid             – process id.
        window_start_ns – nanosecond timestamp of the window's start boundary.
    """
    # Per-PID accumulation buffers
    pid_buf:      Dict[int, List[Dict]] = {}
    pid_procname: Dict[int, str]        = {}

    # Shared entry→exit pairing state (must persist across window boundaries
    # so an entry event in window N can be paired with its exit in window N+1)
    event_time_map: Dict[str, int]            = {}
    event_time_map_hist: Dict[str, List[int]] = {}

    window_start: Optional[int] = None
    window_end:   Optional[int] = None

    def _flush(ws: int):
        """Yield and clear all PID buffers that meet the min_events threshold."""
        for _pid, buf in pid_buf.items():
            if len(buf) < min_events:
                continue
            _proc = pid_procname.get(_pid, "")
            if filter_procs is not None:
                if not any(s in _proc for s in filter_procs):
                    continue
            yield buf, _proc, _pid, ws

    for event in events:
        ts  = event["timestamp"]
        pid = event.get("pid", 0)

        # Compute entry→exit latency (shares state with trace_reader.get_duration)
        lat, f_mean, _err = get_duration(event, event_time_map, event_time_map_hist)
        event["latency"] = lat
        event["f_mean"]  = f_mean

        # Initialise window on first event
        if window_start is None:
            window_start = ts
            window_end   = ts + window_ns

        # Window boundary crossed — flush, then advance
        if ts >= window_end:
            yield from _flush(window_start)

            # Advance window so ts falls inside the new window
            n_skipped    = (ts - window_start) // window_ns
            window_start += n_skipped * window_ns
            window_end    = window_start + window_ns

            # Reset per-PID buffers (latency map persists for pairing)
            pid_buf.clear()

        # Accumulate event into PID buffer
        if pid not in pid_buf:
            pid_buf[pid] = []
        pid_buf[pid].append(event)
        pid_procname[pid] = event.get("procname", "")

    # Final partial window
    if window_start is not None:
        yield from _flush(window_start)


# ---------------------------------------------------------------------------
# Helper: fast trace duration scan (single lightweight pass)
# ---------------------------------------------------------------------------

def scan_trace_duration(trace_path: str) -> Tuple[int, int, int]:
    """Return (first_ts_ns, last_ts_ns, n_events) for a trace.

    Used by the preprocessor to compute time-based train/valid/test splits
    without needing the full event payload.
    """
    try:
        import bt2
    except ImportError:
        raise ImportError("bt2 not available (https://babeltrace.org)")

    _EvType   = bt2._EventMessageConst
    first_ts  = None
    last_ts   = None
    n         = 0

    for msg in bt2.TraceCollectionMessageIterator(trace_path):
        if type(msg) is not _EvType:
            continue
        ts      = msg.default_clock_snapshot.ns_from_origin
        last_ts = ts
        if first_ts is None:
            first_ts = ts
        n += 1

    return first_ts or 0, last_ts or 0, n

"""Optimized LTTng trace reading.

Replicates the logic of ``functions.get_events``, ``functions.get_requests``,
and ``functions.get_duration`` from the original LMAT codebase, with the
following micro-optimisations:

* ``_EventMessageConst`` type is cached once to avoid repeated attribute
  lookups inside the hot loop.
* The ``keys`` dict is converted to a list of tuples before the loop so
  ``dict.items()`` is not called per event.
* Name alias table is a module-level dict (``_NAME_ALIAS``) to avoid
  repeated string-construction.
* A single dict literal is used for the event object instead of empty
  dict + repeated assignment.

The public API is:
    load_trace(file_path)               → bt2.TraceCollectionMessageIterator
    get_events(trace, keys)             → generator of event dicts
    get_requests(events)                → generator of (request, hist, err_count)
    get_duration(event, time_map, hist) → (latency_ns, f_mean_ns, error_count)
"""

import itertools
import sys
from time import time
from datetime import timedelta
from typing import Generator, Dict, Any, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Name alias table (normalises a handful of kernel event names)
# ---------------------------------------------------------------------------
_NAME_ALIAS: Dict[str, str] = {
    "block_rq_insert":       "block_rq_entry",
    "block_rq_complete":     "block_rq_exit",
    "timer_hrtimer_start":   "timer_hrtimer_entry",
    "timer_hrtimer_cancel":  "timer_hrtimer_exit",
    "timer_start":           "timer_entry",
    "timer_cancel":          "timer_exit",
}

# Apache httpd boundary events
_HTTPD_ENTER = "httpd:enter_event_handler"
_HTTPD_EXIT  = "httpd:exit_event_handler"

# Connection states to ignore (lingering connections, not real requests)
_IGNORED_CONN_STATES = {6, 7}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_trace(file_path: str):
    """Open a LTTng CTF trace and return a babeltrace2 message iterator.

    Args:
        file_path: Path to the LTTng trace directory.

    Returns:
        bt2.TraceCollectionMessageIterator ready to iterate.
    """
    try:
        import bt2
    except ImportError:
        raise ImportError(
            "Library bt2 is not available. "
            "See https://babeltrace.org for installation instructions."
        )
    return bt2.TraceCollectionMessageIterator(file_path)


def get_events(
    trace_collection,
    keys: Optional[Dict[str, str]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """Yield kernel events as plain Python dicts.

    This is a speed-optimised version of the original ``get_events``.
    The babeltrace2 C-extension is still the primary bottleneck; what we
    improve is the Python-level overhead around each event:

    * type check uses a cached class reference (no per-iteration attribute
      lookup on the module).
    * ``keys`` dict is pre-converted to a list of (field_name, dest_key) tuples.
    * Name aliasing uses a module-level dict lookup (O(1)) instead of a chain
      of if/elif string comparisons.

    Args:
        trace_collection: A ``bt2.TraceCollectionMessageIterator``.
        keys: Mapping from babeltrace field names to dest dict keys, e.g.
              ``{"vtid": "tid", "pid": "pid", "procname": "procname", ...}``.

    Yields:
        dict with at minimum: name, timestamp, connection_state,
        sector, next_tid, plus whatever is in *keys*.
    """
    try:
        import bt2
    except ImportError:
        raise ImportError("bt2 not available (https://babeltrace.org)")

    # Cache the class once — avoids repeated module attribute lookup in the loop
    _EventMsgType = bt2._EventMessageConst
    keys_items: List[Tuple[str, str]] = list(keys.items()) if keys else []

    for msg in trace_collection:
        if type(msg) is not _EventMsgType:
            continue

        ev      = msg.event
        raw_name = ev.name
        name    = _NAME_ALIAS.get(raw_name, raw_name)  # O(1) alias lookup
        ts      = msg.default_clock_snapshot.ns_from_origin

        # Apache connection_state (only on httpd boundary events)
        if name is _HTTPD_ENTER or name is _HTTPD_EXIT or \
           name == _HTTPD_ENTER or name == _HTTPD_EXIT:
            pf    = ev.payload_field
            conn  = pf["connection_state"]
            conn_state = -1 if conn is None else int(conn)
        else:
            conn_state = -1

        # sector / next_tid (only for specific event types)
        sector   = None
        next_tid = None
        if name == "block_rq_entry" or name == "block_rq_exit":
            sector = ev["sector"]
        elif name == "sched_switch":
            next_tid = ev["next_tid"]

        event: Dict[str, Any] = {
            "name":             name,
            "timestamp":        ts,
            "connection_state": conn_state,
            "sector":           sector,
            "ptr":              None,
            "next_tid":         next_tid,
        }

        # Extract requested fields from the event payload
        for field_name, dest_key in keys_items:
            try:
                event[dest_key] = ev[field_name]
            except KeyError:
                pass  # field not present in this event type

        yield event


# ---------------------------------------------------------------------------

def get_duration(
    event: Dict[str, Any],
    event_time_map: Dict[str, int],
    event_time_map_hist: Dict[str, List[int]],
) -> Tuple[Optional[int], Optional[int], int]:
    """Compute the latency (ns) and rolling-mean latency for one event.

    Logic is identical to the original ``functions.get_duration``.  For
    *entry* events the latency is 0 (no paired exit yet).  For *exit*
    events the latency is ``exit_ts - entry_ts`` and the rolling mean is
    the mean of the last 10 durations for that event type.

    Args:
        event:              The current event dict (must have 'name', 'pid',
                            'tid', 'timestamp').
        event_time_map:     Mutable dict tracking the most-recent entry
                            timestamp for each (event_name, pid, tid) key.
        event_time_map_hist: Mutable dict tracking the list of all durations
                            for each event_name (strip pid/tid).

    Returns:
        (latency_ns, f_mean_ns, error_count)
        latency_ns  – raw duration in nanoseconds (0 for entry events).
        f_mean_ns   – rolling mean of the last 10 durations (0 for entry).
        error_count – 1 if a pairing error was detected, 0 otherwise.
    """
    name = event["name"]
    # Normalise to syscall base name (strip syscall_/entry_/exit_)
    sysname  = name.replace("syscall_", "").replace("entry_", "").replace("exit_", "")
    tmp_id   = sysname + "@" + str(event["pid"]) + str(event["tid"])
    prev_ts  = event_time_map.get(tmp_id)
    cur_ts   = event["timestamp"]
    f_mean   = None
    latency  = None
    err      = 0

    if "entry" in name:
        if prev_ts is None:
            event_time_map[tmp_id] = cur_ts
        elif prev_ts == -1:
            event_time_map[tmp_id] = cur_ts
        else:
            err += 1
        f_mean  = 0
        latency = 0

    elif "exit" in name:
        if prev_ts is None or prev_ts == -1:
            f_mean  = 0
            latency = 0
            event_time_map[tmp_id] = -1
            err += 1
        else:
            event_time_map[tmp_id] = -1
            hist_key = sysname  # strip pid/tid: history is per event type
            latency  = cur_ts - prev_ts
            hist     = event_time_map_hist.get(hist_key)
            if hist is None:
                event_time_map_hist[hist_key] = [latency]
                f_mean = 0
            else:
                f_mean = sum(hist[-10:]) / min(len(hist), 10)
                hist.append(latency)

    return latency, f_mean, err


# ---------------------------------------------------------------------------

def get_requests(
    events: Generator[Dict[str, Any], None, None],
) -> Generator[Tuple[List[Dict], Dict, int], None, None]:
    """Split a stream of kernel events into per-request event lists.

    Identical logic to the original ``functions.get_requests``.

    A request starts when an ``httpd:enter_event_handler`` event is seen
    (for calls whose ``connection_state`` is not 6 or 7) and ends at the
    matching ``httpd:exit_event_handler`` event.

    Yields:
        (request_events, event_time_map_hist, error_count)
        request_events    – ordered list of event dicts for one request.
        event_time_map_hist – rolling latency history dict at request end.
        error_count       – number of entry/exit pairing errors seen.
    """
    threads: Dict[int, List[Dict]]      = {}  # tid → [events]
    event_time_map: Dict[str, int]      = {}
    event_time_map_hist: Dict[str, List[int]] = {}
    error_count = 0

    for event in events:
        latency_val, f_mean_val, err = get_duration(
            event, event_time_map, event_time_map_hist
        )
        name = event["name"]

        if name == _HTTPD_ENTER:
            if event["connection_state"] not in _IGNORED_CONN_STATES:
                threads[event["tid"]] = []

        elif name == _HTTPD_EXIT:
            tid = event["tid"]
            if tid in threads:
                if threads[tid]:
                    yield threads[tid], event_time_map_hist, error_count
                    error_count = 0
                del threads[tid]

        else:
            event["latency"] = latency_val
            event["f_mean"]  = f_mean_val
            error_count      += err
            for req_buf in threads.values():
                req_buf.append(event)

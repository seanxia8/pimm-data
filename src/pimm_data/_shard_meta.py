"""Cached per-shard metadata reader (Phase A / A1).

At construction, several readers open the same shard file more than once —
each reader scans its shards for the present ``event_*`` groups, and the
sensor reader opens every file again just to detect the readout type. At
doraemon scale (hundreds of shards × 10–30 ms cold each on SDF) that
redundancy is the dominant ``__init__`` cost.

This memoizes the cheap metadata scan — ``n_events``, the present
event-number set, and the ``config`` attrs — keyed on ``(path, mtime,
size)`` so a rewritten file busts the cache (important for tests that edit a
fixture in place). A cache hit opens nothing. Only successful reads are
cached; an open failure propagates so the caller can fall back.
"""

import os

import numpy as np
import h5py

_CACHE = {}


def read_shard_meta(path):
    """Return ``{n_events, present_events, config_attrs}`` for one shard.

    ``present_events`` is a sorted ``np.int64`` array of the event numbers
    actually present (gap-tolerant). Cached by ``(path, mtime_ns, size)``.
    The returned arrays/dicts are shared cache values — treat as read-only.
    """
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    with h5py.File(path, 'r', libver='latest', swmr=True) as f:
        config_attrs = dict(f['config'].attrs) if 'config' in f else {}
        present = np.array(sorted(
            int(k.rsplit('_', 1)[1]) for k in f.keys()
            if k.startswith('event_')), dtype=np.int64)
    meta = {
        'n_events': int(config_attrs.get('n_events', len(present))),
        'present_events': present,
        'config_attrs': config_attrs,
    }
    _CACHE[key] = meta
    return meta


def clear_cache():
    """Drop all memoized shard metadata (test isolation / freed handles)."""
    _CACHE.clear()

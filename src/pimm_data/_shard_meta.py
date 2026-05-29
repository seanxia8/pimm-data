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
_DEPOSIT_CACHE = {}


def read_deposit_counts(path):
    """Per-present-event deposit counts for the ``min_deposits`` filter (F16).

    Returns ``{event_num: {'per_vol': [n_actual per volume id], 'positions':
    int, 'n_volumes': int}}``, cached by ``(path, mtime_ns, size)``. ``per_vol``
    is indexed by volume id (0 for an absent volume), so a ``volume=`` filter
    selects ``per_vol[volume]`` correctly.

    Reading per-event ``n_actual`` attrs is the dominant index-build cost when
    ``min_deposits>0`` (≈75 ms/file on SDF). Memoizing it lets train/val/test
    (and tiered medium/high) datasets built over the same shards share one
    scan instead of repeating it per construction.
    """
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    hit = _DEPOSIT_CACHE.get(key)
    if hit is not None:
        return hit
    counts = {}
    with h5py.File(path, 'r', libver='latest', swmr=True) as f:
        n_volumes = (int(f['config'].attrs.get('n_volumes', 1))
                     if 'config' in f else 1)
        for k in f.keys():
            if not k.startswith('event_'):
                continue
            num = int(k.rsplit('_', 1)[1])
            evt = f[k]
            per_vol = [
                (int(evt[f'volume_{v}'].attrs.get('n_actual', 0))
                 if f'volume_{v}' in evt else 0)
                for v in range(n_volumes)]
            pos = evt['positions'].shape[0] if 'positions' in evt else 0
            counts[num] = {'per_vol': per_vol, 'positions': pos,
                           'n_volumes': n_volumes}
    _DEPOSIT_CACHE[key] = counts
    return counts


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
        cfg = f['config'] if 'config' in f else None
        config_attrs = dict(cfg.attrs) if cfg is not None else {}
        present = np.array(sorted(
            int(k.rsplit('_', 1)[1]) for k in f.keys()
            if k.startswith('event_')), dtype=np.int64)
        # Per-file stable-identity vector (O(1)/file), if the writer stamped
        # it; indexable by event number. Falls back to None (the dataset then
        # uses the per-event attr or positional identity — D26).
        sei_vec = None
        if cfg is not None and 'source_event_idx' in cfg:
            sei_vec = cfg['source_event_idx'][()].astype(np.int64)
    meta = {
        'n_events': int(config_attrs.get('n_events', len(present))),
        'present_events': present,
        'config_attrs': config_attrs,
        'source_event_idx': sei_vec,
        # Identity inputs (F1): file_index is the intrinsic shard id (both
        # detectors); global_event_offset lets a shard-local per-event
        # source_event_idx be resolved without opening event groups.
        'global_event_offset': (int(config_attrs['global_event_offset'])
                                 if 'global_event_offset' in config_attrs
                                 else None),
        'file_index': (int(config_attrs['file_index'])
                       if 'file_index' in config_attrs else None),
    }
    _CACHE[key] = meta
    return meta


def clear_cache():
    """Drop all memoized shard metadata (test isolation / freed handles)."""
    _CACHE.clear()
    _DEPOSIT_CACHE.clear()

"""Generic label decoration (Part 04 / D20/D22/D28/D38).

The datasets attach per-point labels to a modality by gathering values from the
``labl`` tables through a per-point foreign key. Historically each dataset
hand-wrote this (one bare ``segment`` + ``instance`` axis). This module
factors out the gather and drives an **open, multi-axis** schema from a
declarative ``label_config`` so a dataset can emit named keys
(``segment_pid``/``instance_particle``/``instance_interaction``/``target_*``)
without bespoke code per axis.

Two gather kinds, selected by whether the labl table is positionally indexed
(LUCiD ``per_particle``/``per_track`` — ``keyed_by=None``) or value-keyed
(JAXTPC ``track_ids`` — ``keyed_by`` is the key column). The only
detector-specific piece is the ``fk_resolver`` (which per-point FK feeds an
axis); everything else is shared.

A ``label_config`` entry is a dict:

* ``out``    — emitted key (``segment_pid``, ``instance_particle``, …).
* ``scope``  — ``'point'`` (per-point gather; default), ``'event'`` (one
  value per event, attached as-is), or ``'event_broadcast'`` (one value tiled
  to every point).
* ``fk``     — name passed to ``fk_resolver`` (point scope only).
* ``source`` — ``'self'`` (the FK value *is* the label) or ``(table, column)``
  into the nested labl dict.
* ``keyed_by`` — optional key column name in ``table`` for value-keyed gather.
* ``fill``   — sentinel for unresolved FKs (default ``-1``).
"""

import numpy as np


def gather_with_fill(fk, column, keyed_by=None, fill=-1):
    """Per-point gather of ``column`` for each foreign key in ``fk``.

    ``keyed_by=None`` → positional index (``column[fk]`` with a bounds mask).
    Otherwise ``keyed_by`` is the value table that ``fk`` indexes by value
    (searchsorted, match-verified) — the JAXTPC ``track_ids`` case.
    """
    fk = np.asarray(fk)
    # Output dtype must hold both the column values and the (-1) fill sentinel.
    # A bool/unsigned column would make `fill=-1` silently True (bool) or
    # OverflowError (uint) — real labl `category` is uint8, `contained` is bool.
    dt = column.dtype
    if not (np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.signedinteger)):
        dt = np.int64
    out = np.full(fk.shape, fill, dtype=dt)
    if fk.size == 0 or column.shape[0] == 0:
        return out
    if keyed_by is None:
        valid = (fk >= 0) & (fk < column.shape[0])
        if valid.any():
            out[valid] = column[fk[valid]]
        return out
    order = np.argsort(keyed_by)
    s_keys = keyed_by[order]
    s_vals = column[order]
    pos = np.clip(np.searchsorted(s_keys, fk), 0, len(s_keys) - 1)
    matched = s_keys[pos] == fk
    out[matched] = s_vals[pos[matched]]
    return out


def _labl_column(labl, source):
    """Resolve ``source`` to a labl column array (or None if absent)."""
    if not isinstance(source, (tuple, list)) or len(source) != 2:
        raise ValueError(f"label source must be (table, column), got {source!r}")
    table, column = source
    tbl = labl.get(table) if labl else None
    if tbl is None:
        return None
    return tbl.get(column)


def decorate_labels(sub, labl, fk_resolver, label_config):
    """Attach the named schema keys in ``label_config`` to a modality sub-dict.

    Parameters
    ----------
    sub : dict
        Stream sub-dict (has ``coord`` and the per-point FK columns).
    labl : dict
        Nested ``{table: {column: array}}`` label tables for this event.
    fk_resolver : callable
        ``fk_name -> per-point int array`` (or ``None`` if unavailable).
    label_config : list[dict]
        Axis specs (see module docstring).
    """
    if labl is None or label_config is None:
        return sub
    n = sub['coord'].shape[0]
    for spec in label_config:
        out = spec['out']
        scope = spec.get('scope', 'point')

        if scope in ('event', 'event_broadcast'):
            val = _labl_column(labl, spec['source'])
            if val is None:
                continue
            val = np.asarray(val)
            if scope == 'event_broadcast':
                # tile a single per-event value to every point
                flat = val.reshape(-1)
                sub[out] = np.repeat(flat[None, :], n, axis=0) if flat.size > 1 \
                    else np.full((n, 1), flat[0])
            else:
                sub[out] = val
            continue

        fk = fk_resolver(spec['fk'])
        if fk is None:
            continue
        if spec.get('source', 'self') == 'self':
            sub[out] = np.asarray(fk).astype(np.int32)[:, None]
            continue
        col = _labl_column(labl, spec['source'])
        if col is None:
            continue
        keyed_by = None
        if spec.get('keyed_by') is not None:
            keyed_by = _labl_column(labl, (spec['source'][0], spec['keyed_by']))
        g = gather_with_fill(fk, col, keyed_by=keyed_by,
                             fill=spec.get('fill', -1))
        sub[out] = g[:, None] if g.ndim == 1 else g
    return sub

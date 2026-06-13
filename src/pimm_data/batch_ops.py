"""Boundary helpers for the flat-prefixed batch (REDESIGN.md §6).

The batch is a plain dict of tensors (+ a carried ``_roles`` map). These helpers do
the load-bearing reconstructions ONCE, so model/training code never hand-rolls them:

* :func:`to_batched_coords` — prepend the ``[batch_id, …]`` column sparse-conv
  (MinkowskiEngine / spconv) requires; there is no offset-only path into those.
* :func:`split_event` — extract one event, with index columns (edges) **rebased** to
  that event and per-point rows sliced — hand-rolled rebasing is the #1 bug source.

Both read ``<part>_offset`` (``(B,)`` cumulative, NO leading 0) and the carried
``_roles`` to know each key's role.
"""

from __future__ import annotations

import torch

from . import _roles


def _roles_of(batch):
    return batch.get('_roles', {})


def to_batched_coords(batch, part, coord_key=None):
    """``[N, 1+D]`` coords with a prepended batch-id column, for sparse-conv.

    ``coord_key`` defaults to ``<part>_coord``. batch_id is derived from
    ``<part>_offset`` (no-leading-0) via :func:`_roles.offset_to_batch`.
    """
    coord_key = coord_key or f'{part}_coord'
    coord = batch[coord_key]
    bid = _roles.offset_to_batch(batch[f'{part}_offset']).to(coord.device)
    return torch.cat([bid.to(coord.dtype).unsqueeze(1), coord], dim=1)


def split_event(batch, i):
    """One event ``i`` as a flat dict, with every part's rows sliced and index
    arrays (edges) **rebased** to start at 0 for this event.

    Whole-event (unprefixed / ``event``-role) keys are indexed at ``i``; ``_roles``
    is carried (with the same specs). Requires the carried ``_roles`` for edge/event.
    """
    roles = _roles_of(batch)
    keys = [k for k in batch if k != '_roles']
    parts = _roles.parts_from_keys(keys, roles)
    sub_offsets = _roles.subspace_offset_keys(keys, roles)

    # per-part [lo, hi) POINT span for event i (sliced by <part>_offset).
    span = {}
    for p in parts:
        off = batch[f'{p}_offset']
        lo = int(off[i - 1]) if i > 0 else 0
        span[p] = (lo, int(off[i]))

    # per-second-row-space [lo, hi) span, keyed by the offset KEY. A part with a
    # second row-space (REDESIGN §3) carries instance-role keys (bbox K rows;
    # packed waveform samples) counted by their own offset, NOT by the point
    # count — they must slice by this span, never the point span.
    sub_span = {}
    for k in sub_offsets:
        off = batch[k]
        lo = int(off[i - 1]) if i > 0 else 0
        sub_span[k] = (lo, int(off[i]))

    out = {}
    for key in keys:
        spec = roles.get(key)
        kind = _roles.role_kind(spec) if spec is not None else None
        if key in sub_offsets:                              # 2nd-row-space offset
            lo, hi = sub_span[key]
            out[key] = batch[key].new_tensor([hi - lo])     # single-event count (1,)
            continue
        if key.endswith('_offset'):                         # part (point) offset
            lo, hi = span.get(key[:-len('_offset')], (0, 0))
            out[key] = batch[key].new_tensor([hi - lo])     # single-event offset (1,)
            continue
        if kind == 'instance':                              # slice by its own row-space
            lo, hi = sub_span.get(spec[1], (0, 0))
            out[key] = batch[key][lo:hi]
            continue
        if kind == 'edge':
            tgt = spec[1]
            v = batch[key]
            if tgt == 'self':
                p = _roles.part_of(key, parts)
                # keep edges whose endpoints are in this event's span, rebase to 0
                lo, hi = span[p]
                m = (v[0] >= lo) & (v[0] < hi) & (v[1] >= lo) & (v[1] < hi)
                out[key] = v[:, m] - lo
            else:
                src, dst = tgt
                (slo, shi), (dlo, dhi) = span[src], span[dst]
                m = (v[0] >= slo) & (v[0] < shi) & (v[1] >= dlo) & (v[1] < dhi)
                e = v[:, m].clone()
                e[0] -= slo
                e[1] -= dlo
                out[key] = e
            continue
        if kind == _roles.EVENT or _roles.part_of(key, parts) is None:
            out[key] = batch[key][i]                        # stacked (B,…) or list -> event i
            continue
        # point / raw / label -> slice this part's POINT rows
        p = _roles.part_of(key, parts)
        lo, hi = span[p]
        out[key] = batch[key][lo:hi]
    if roles:
        out['_roles'] = roles
    return out

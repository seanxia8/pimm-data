"""Per-part roles + role-driven batching for the flat-prefixed batch contract.

A *part* is a set of flat, underscore-prefixed keys (``step_coord``, ``step_offset``)
sharing a ``<part>_offset`` (cumulative per-event counts, ``(B,)`` **NO leading 0**,
``offset[-1] == ΣN`` — the existing convention; helpers prepend the 0 internally).
A sample/batch may carry a ``_roles`` map naming the **non-default** keys; everything
unlisted (and matching its part's point count) is ``point``. Roles drive collate
(this module) and subsampling (:func:`pimm_data.transform.index_operator`).

Role spec values (what ``_roles[key]`` holds):

==================  =========================================================
spec                meaning / collate behaviour
==================  =========================================================
``'point'``         one row per point — concat by ``offset`` (the default; rarely listed)
``'raw'``           per-point but immutable (densify COO) — concat by ``offset``; never sliced
``'event'``         NOT per-point (whole-event scalar / part summary / dense grid) — **stack** ``(B,…)`` / list
``('edge','self')`` index into its own part — concat + shift by the part's running node count
``('edge',(s,d))``  bipartite cross-store — row 0 shifts by ``s``'s node count, row 1 by ``d``'s
``('instance',ok)`` one row per instance — concat by ``<part>_offset`` named ``ok`` (an ``*_inst_offset`` key)
``('label',grp)``   categorical id — compact per event to ``0..K-1`` then add running distinct-count base, joint over ``grp``
==================  =========================================================

This module is import-light (torch only) and has no pimm-data dependencies, so it can
be reused by collate and by the post-collate helpers.
"""

from __future__ import annotations

import torch

POINT = 'point'
RAW = 'raw'
EVENT = 'event'


def offset_to_batch(offset):
    """``(B,)`` cumulative offset (NO leading 0) -> per-row batch index ``(ΣN,)``.

    Prepends the 0 internally (``diff(prepend=0)``); empty events contribute nothing.
    """
    counts = torch.diff(offset, prepend=offset.new_zeros(1))
    return torch.repeat_interleave(
        torch.arange(offset.numel(), device=offset.device), counts)


def node_bases(offset):
    """Per-event running node base from a ``(B,)`` offset: ``[0, n0, n0+n1, …]`` (B,).

    The amount to add to event ``i``'s within-event indices so concatenated index
    arrays stay globally valid. = the offset shifted right by one with a leading 0.
    """
    return torch.cat([offset.new_zeros(1), offset[:-1]])


# ── role parsing ─────────────────────────────────────────────────────────────
def role_kind(spec):
    """Normalize a role spec to its kind string ('point'|'raw'|'event'|'edge'|
    'instance'|'label')."""
    if spec in (POINT, RAW, EVENT):
        return spec
    if isinstance(spec, (tuple, list)) and spec:
        return spec[0]
    raise ValueError(f"unknown role spec: {spec!r}")


def parts_from_keys(keys):
    """The set of part names present = prefixes ``P`` such that ``P_offset`` is a key.

    ``*_inst_offset`` keys name an instance sub-offset of part ``P`` (``P_inst``),
    not a separate part — exclude them from the part set.
    """
    suf = '_offset'
    out = set()
    for k in keys:
        if k.endswith(suf):
            p = k[:-len(suf)]
            if p.endswith('_inst'):
                continue
            out.add(p)
    return out


def part_of(key, parts):
    """Longest-prefix-match: the part a flat key belongs to (or None = whole-event)."""
    cands = [p for p in parts if key == p or key.startswith(p + '_')]
    return max(cands, key=len) if cands else None


# ── per-role batch ops (operate on a LIST of per-sample tensors) ──────────────
def cat_point(parts):
    """point/raw: concatenate along dim 0."""
    return torch.cat(list(parts), dim=0)


def cat_offset(offsets):
    """offset: per-event counts -> cumulative (B,), no leading 0.

    Each sample's offset is its own ``(b_i,)`` cumulative; recover counts via diff,
    concat, cumsum — matches the existing collate convention exactly.
    """
    counts = [o.diff(prepend=o.new_zeros(1)) for o in offsets]
    return torch.cumsum(torch.cat(counts, dim=0), dim=0)


def cat_edge_self(edges, offsets):
    """('edge','self'): shift each event's edges by its node base, then concat (dim -1)."""
    return _shift_concat(edges, offsets, dst_offsets=None)


def _node_counts(offsets):
    """Per-sample node count from each sample's own (b_i,) offset = off[-1] (or 0)."""
    return [int(o[-1]) if o.numel() else 0 for o in offsets]


def _shift_concat(edges, src_offsets, dst_offsets=None):
    """Concat edge_index arrays (2,E) shifting row 0 by running src node base and
    row 1 by running dst node base (dst defaults to src for 'self')."""
    if dst_offsets is None:
        dst_offsets = src_offsets
    src_counts = _node_counts(src_offsets)
    dst_counts = _node_counts(dst_offsets)
    src_base = dst_base = 0
    out = []
    for i, e in enumerate(edges):
        e = e.clone()
        e[0] = e[0] + src_base
        e[1] = e[1] + dst_base
        out.append(e)
        src_base += src_counts[i]
        dst_base += dst_counts[i]
    return torch.cat(out, dim=-1)


def cat_label_col(values):
    """('label', …): one categorical id column across the batch.

    Compact each event's ids to ``0..K-1`` (raw FK ids aren't dense), then add a
    running distinct-count base. Applied independently per column, this preserves
    the hierarchy (a cluster still maps to one group: within an event the
    cluster→group map is intact, and both columns shift by their own running
    distinct-count, so the global map stays one-to-one).
    """
    base = 0
    out = []
    for v in values:                          # one event
        uniq, inv = torch.unique(v, return_inverse=True)
        out.append(inv + base)
        base += int(uniq.numel())
    return torch.cat(out, dim=0)


def stack_event(vals):
    """event: stack tensors -> (B,...); list non-tensors."""
    if isinstance(vals[0], torch.Tensor):
        return torch.stack(list(vals), dim=0)
    return list(vals)

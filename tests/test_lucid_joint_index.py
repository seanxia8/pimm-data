"""Phase A regression tests for LUCiD cross-modality event alignment.

LUCiD has the same desync shape as JAXTPC: the dataset passed one global
``idx`` to every reader with ``_n_events = min(len(r))``, so ``min_segments``
(which masks the step reader's index only) silently misaligned step against
sensor/hits/labl. ``build_joint_index`` intersects the present events across
modalities and injects one shared index, keeping all modalities on the same
physics event for every idx.

The LUCiD readers index ``read_shard_meta(...)['present_events']`` (F6), so —
like JAXTPC — a missing ``event_NNN`` is skipped per modality and the joint
index intersects what each modality actually has. ``min_segments`` is the
other desync source (it masks the step index only).
"""
import os

import h5py
import pytest

from pimm_data.testing import make_lucid_sample
from pimm_data.lucid import LUCiDDataset

from _joint_index_helpers import readers as _readers, assert_aligned as _assert_aligned

_ALL = ('step', 'sensor', 'hits', 'labl')


def test_no_filter_all_modalities_aligned(tmp_path):
    root = make_lucid_sample(str(tmp_path), n_events=3)
    ds = LUCiDDataset(data_root=root, split='', modalities=_ALL)
    assert len(ds) == 3
    _assert_aligned(ds)
    for idx in range(len(ds)):
        ds.get_data(idx)


def test_min_segments_desync_realigns(tmp_path):
    """min_segments drops a low-segment step event; joint drops it everywhere."""
    root = make_lucid_sample(str(tmp_path), n_events=3)
    # event_001 reports 0 segments → step filters it; others still hold it.
    with h5py.File(os.path.join(root, 'step', 'wc_step_0000.h5'), 'r+') as f:
        f['event_001'].attrs['n_segments'] = 0

    ds = LUCiDDataset(data_root=root, split='', modalities=_ALL,
                      min_segments=1)
    assert len(ds) == 2
    for r in _readers(ds):
        assert r.indices[0].tolist() == [0, 2]
    _assert_aligned(ds)
    for idx in range(len(ds)):
        ds.get_data(idx)


def test_missing_event_group_gap_tolerant(tmp_path):
    """F6: a deleted ``event_NNN`` in one modality is skipped (not crashed on),
    and the joint index drops it everywhere so modalities stay aligned.

    Pre-F6 the readers used ``arange(n_events)`` from the (unchanged) config
    attr, so a punched gap meant opening a missing group + off-by-one misalign
    against every other modality. ``present_events`` makes the skip intrinsic.
    """
    root = make_lucid_sample(str(tmp_path), n_events=4)
    # Punch an interior gap in hits only (config n_events still says 4).
    with h5py.File(os.path.join(root, 'hits', 'wc_hits_0000.h5'), 'r+') as f:
        del f['event_002']

    ds = LUCiDDataset(data_root=root, split='', modalities=_ALL)
    assert len(ds) == 3                              # 002 gone, not crashed
    for r in _readers(ds):
        assert r.indices[0].tolist() == [0, 1, 3]    # gap-tolerant + intersected
    _assert_aligned(ds)
    for idx in range(len(ds)):
        ds.get_data(idx)


def test_min_segments_without_step_raises(tmp_path):
    root = make_lucid_sample(str(tmp_path), n_events=2)
    with pytest.raises(ValueError, match="min_segments"):
        LUCiDDataset(data_root=root, split='', modalities=('hits', 'labl'),
                     min_segments=5)


def test_strict_lengths_raises_on_cross_modality_drop(tmp_path):
    root = make_lucid_sample(str(tmp_path), n_events=3)
    with h5py.File(os.path.join(root, 'step', 'wc_step_0000.h5'), 'r+') as f:
        f['event_001'].attrs['n_segments'] = 0
    with pytest.raises(ValueError):
        LUCiDDataset(data_root=root, split='', modalities=_ALL,
                     min_segments=1, strict_lengths=True)
    # Default (non-strict) warns and proceeds.
    ds = LUCiDDataset(data_root=root, split='', modalities=_ALL,
                      min_segments=1)
    assert len(ds) == 2

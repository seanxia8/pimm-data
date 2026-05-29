"""Phase A regression tests for LUCiD cross-modality event alignment.

LUCiD has the same desync shape as JAXTPC: the dataset passed one global
``idx`` to every reader with ``_n_events = min(len(r))``, so ``min_segments``
(which masks the edep reader's index only) silently misaligned edep against
sensor/hits/labl. ``build_joint_index`` intersects the present events across
modalities and injects one shared index, keeping all modalities on the same
physics event for every idx.

(Unlike JAXTPC, the LUCiD readers index ``arange(n_events)`` — they are not
gap-tolerant — so the LUCiD desync source is ``min_segments``, not a missing
``event_NNN``.)
"""
import os

import h5py
import pytest

from pimm_data.testing import make_lucid_sample
from pimm_data.lucid import LUCiDDataset

_ALL = ('edep', 'sensor', 'hits', 'labl')


def _readers(ds):
    return [r for r in (ds.edep_reader, ds.sensor_reader,
                        ds.hits_reader, ds.labl_reader) if r is not None]


def _assert_aligned(ds):
    readers = _readers(ds)
    ref_idx = [a.tolist() for a in readers[0].indices]
    ref_cum = readers[0].cumulative_lengths.tolist()
    for r in readers[1:]:
        assert [a.tolist() for a in r.indices] == ref_idx
        assert r.cumulative_lengths.tolist() == ref_cum
    for r in readers:
        if not r._initted:
            r.h5py_worker_init()
    for idx in range(len(ds)):
        keys = {r._locate_event(idx)[1] for r in readers}
        assert len(keys) == 1, f"idx {idx}: modalities disagree {keys}"


def test_no_filter_all_modalities_aligned(tmp_path):
    root = make_lucid_sample(str(tmp_path), n_events=3)
    ds = LUCiDDataset(data_root=root, split='', modalities=_ALL)
    assert len(ds) == 3
    _assert_aligned(ds)
    for idx in range(len(ds)):
        ds.get_data(idx)


def test_min_segments_desync_realigns(tmp_path):
    """min_segments drops a low-segment edep event; joint drops it everywhere."""
    root = make_lucid_sample(str(tmp_path), n_events=3)
    # event_001 reports 0 segments → edep filters it; others still hold it.
    with h5py.File(os.path.join(root, 'edep', 'wc_edep_0000.h5'), 'r+') as f:
        f['event_001'].attrs['n_segments'] = 0

    ds = LUCiDDataset(data_root=root, split='', modalities=_ALL,
                      min_segments=1)
    assert len(ds) == 2
    for r in _readers(ds):
        assert r.indices[0].tolist() == [0, 2]
    _assert_aligned(ds)
    for idx in range(len(ds)):
        ds.get_data(idx)


def test_min_segments_without_edep_raises(tmp_path):
    root = make_lucid_sample(str(tmp_path), n_events=2)
    with pytest.raises(ValueError, match="min_segments"):
        LUCiDDataset(data_root=root, split='', modalities=('hits', 'labl'),
                     min_segments=5)


def test_strict_lengths_raises_on_cross_modality_drop(tmp_path):
    root = make_lucid_sample(str(tmp_path), n_events=3)
    with h5py.File(os.path.join(root, 'edep', 'wc_edep_0000.h5'), 'r+') as f:
        f['event_001'].attrs['n_segments'] = 0
    with pytest.raises(ValueError):
        LUCiDDataset(data_root=root, split='', modalities=_ALL,
                     min_segments=1, strict_lengths=True)
    # Default (non-strict) warns and proceeds.
    ds = LUCiDDataset(data_root=root, split='', modalities=_ALL,
                      min_segments=1)
    assert len(ds) == 2

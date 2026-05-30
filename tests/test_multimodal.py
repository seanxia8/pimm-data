"""MultiModalEventDataset: source mixture + deterministic hash holdout.

Wraps JAXTPCDataset per source. The headline property: the holdout is keyed on
(config_id, source_event_idx), so it is reproducible and invariant to how
events are sharded — the fix for the positional-permutation flaw (SR5/D26).
"""
import os

import numpy as np
import h5py
import pytest

from pimm_data.testing import make_jaxtpc_sample, make_lucid_sample
from pimm_data.multimodal import MultiModalEventDataset

_SRC = dict(type='JAXTPCDataset', modalities=('edep',), dataset_name='sim')


def _two_sources(tmp_path, n_events=12):
    a = make_jaxtpc_sample(str(tmp_path / 'cfgA'), n_events=n_events)
    b = make_jaxtpc_sample(str(tmp_path / 'cfgB'), n_events=n_events)
    return [dict(root=a, label=0, config_id=0),
            dict(root=b, label=1, config_id=1)]


def _identity_set(ds):
    return {ds.event_identity(i) for i in range(len(ds))}


# ---------------------------------------------------------------------------
# Mixture + labels
# ---------------------------------------------------------------------------

def test_single_source_no_holdout(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=8)
    ds = MultiModalEventDataset(_SRC, [dict(root=root, label=3, config_id=7)])
    assert len(ds) == 8
    sample = ds.get_data(0)
    assert 'edep' in sample
    assert sample['event_label'].tolist() == [3]
    assert sample['config_id'].tolist() == [7]
    # per-point broadcast into the stream
    n = sample['edep']['coord'].shape[0]
    assert sample['edep']['event_label'].shape == (n, 1)
    assert (sample['edep']['event_label'] == 3).all()


def test_mixture_spans_sources_with_distinct_labels(tmp_path):
    ds = MultiModalEventDataset(_SRC, _two_sources(tmp_path, n_events=10))
    assert len(ds) == 20
    labels = {int(ds.get_data(i)['event_label'][0]) for i in range(len(ds))}
    assert labels == {0, 1}


def test_balanced_mixture_replicates_smaller_source(tmp_path):
    a = make_jaxtpc_sample(str(tmp_path / 'a'), n_events=10)
    b = make_jaxtpc_sample(str(tmp_path / 'b'), n_events=2)
    ds = MultiModalEventDataset(
        _SRC, [dict(root=a, label=0, config_id=0),
               dict(root=b, label=1, config_id=1)],
        mixture={'weights': 'balanced'})
    counts = {0: 0, 1: 0}
    for i in range(len(ds)):
        counts[int(ds.get_data(i)['event_label'][0])] += 1
    # small source (2) replicated ~5x to ~match the large (10).
    assert counts[1] >= 8


# ---------------------------------------------------------------------------
# Holdout: determinism, 3-way partition, shard-invariance
# ---------------------------------------------------------------------------

def test_holdout_three_way_partitions_all(tmp_path):
    srcs = _two_sources(tmp_path, n_events=20)
    ho = dict(seed=0, fractions=(0.6, 0.2, 0.2))
    kw = dict(holdout=ho)
    tr = MultiModalEventDataset(_SRC, srcs, split='train', **kw)
    va = MultiModalEventDataset(_SRC, srcs, split='val', **kw)
    te = MultiModalEventDataset(_SRC, srcs, split='test', **kw)
    al = MultiModalEventDataset(_SRC, srcs, split='all', **kw)
    s_tr, s_va, s_te, s_al = map(_identity_set, (tr, va, te, al))
    # disjoint
    assert s_tr & s_va == set()
    assert s_tr & s_te == set()
    assert s_va & s_te == set()
    # union == all
    assert s_tr | s_va | s_te == s_al
    assert len(s_al) == 40


def test_holdout_deterministic(tmp_path):
    srcs = _two_sources(tmp_path, n_events=15)
    ho = dict(seed=42, fractions=(0.7, 0.15, 0.15))
    a = MultiModalEventDataset(_SRC, srcs, split='val', holdout=ho)
    b = MultiModalEventDataset(_SRC, srcs, split='val', holdout=ho)
    assert a.data_list == b.data_list


def test_holdout_invariant_to_shard_add_remove(tmp_path):
    """Removing a shard does NOT change the split of the surviving events.

    Identity = (config_id, file_index, source_event_idx); file_index and
    source_event_idx are intrinsic (stamped in config), so events in the kept
    shards hash identically whether or not other shards are present. (A
    positional permutation, or a (config_id, sei) key on shard-local sei,
    would NOT have this property — the real-data failure mode this fixes.)
    """
    # 3-shard fixture vs the same first-2 shards (same seed → byte-identical
    # files 0,1; the global source_event_idx + file_index are stamped intrinsic).
    a = make_jaxtpc_sample(str(tmp_path / 'a'), n_events=6, n_files=3)
    b = make_jaxtpc_sample(str(tmp_path / 'b'), n_events=6, n_files=2)
    ho = dict(seed=1, fractions=(0.6, 0.2, 0.2))
    da = MultiModalEventDataset(
        _SRC, [dict(root=a, label=0, config_id=0)], split='train', holdout=ho)
    db = MultiModalEventDataset(
        _SRC, [dict(root=b, label=0, config_id=0)], split='train', holdout=ho)
    a_train = _identity_set(da)
    b_train = _identity_set(db)
    # b is a's first two shards → b's train set == a's train events in files 0,1.
    assert b_train == {idn for idn in a_train if idn[1] in (0, 1)}
    assert len(b_train) > 0


def test_event_identity_unique_no_collisions(tmp_path):
    """Identity is UNIQUE across a multi-shard source (the F1 bug: shard-local
    source_event_idx collided across shards)."""
    root = make_jaxtpc_sample(str(tmp_path), n_events=8, n_files=3)
    ds = MultiModalEventDataset(_SRC, [dict(root=root, label=0, config_id=0)],
                                split='all')
    ids = [ds.event_identity(i) for i in range(len(ds))]
    assert len(ids) == 24
    assert len(set(ids)) == len(ids)        # no collisions across 3 shards


def test_n_per_config_holdout(tmp_path):
    srcs = _two_sources(tmp_path, n_events=20)
    ho = dict(seed=0, n_per_config=4)
    tr = MultiModalEventDataset(_SRC, srcs, split='train', holdout=ho)
    ho_ds = MultiModalEventDataset(_SRC, srcs, split='val', holdout=ho)
    # 4 held out per source × 2 sources = 8 held out; 40 - 8 = 32 train.
    assert len(ho_ds) == 8
    assert len(tr) == 32
    assert _identity_set(tr) & _identity_set(ho_ds) == set()


def test_min_deposits_filter_through_composition(tmp_path):
    """Event filtering works via the source_dataset's own min_deposits
    (joint-aligned by Phase A) — the base wraps the filtered sub-dataset."""
    root = make_jaxtpc_sample(str(tmp_path), n_events=4, n_volumes=2)
    # zero event_001's deposits → min_deposits drops it from the sub-dataset.
    with h5py.File(os.path.join(root, 'edep', 'sim_edep_0000.h5'), 'r+') as f:
        for vk in [k for k in f['event_001'] if k.startswith('volume_')]:
            f['event_001'][vk].attrs['n_actual'] = 0
    src = dict(type='JAXTPCDataset', modalities=('edep',),
               dataset_name='sim', min_deposits=1)
    ds = MultiModalEventDataset(src, [dict(root=root, label=0, config_id=0)])
    assert len(ds) == 3                       # event_001 filtered out
    # the dropped event is absent from identity too (sei is element 2)
    seis = {idn[2] for idn in _identity_set(ds)}
    assert 1 not in seis


def test_min_points_filters_low_pmt_events(tmp_path):
    """min_points (LUCiD SSL dissolve): events with too few sensor PMTs are
    dropped at index time (strict >), counting unique PMTs."""
    root = make_lucid_sample(str(tmp_path), n_events=6, n_sensors=64, n_hits=120)
    src = dict(type='LUCiDDataset', modalities=('sensor',), dataset_name='wc')
    sources = [dict(root=root, label=0, config_id=0)]
    base = MultiModalEventDataset(src, sources, split='all')
    n_all = len(base)
    assert n_all == 6
    # unique PMTs per event <= n_sensors=64, so min_points=64 drops every event
    with pytest.raises(ValueError, match='0 events'):
        MultiModalEventDataset(src, sources, split='all', min_points=64)
    # a low threshold keeps (almost) all; never more than the baseline
    kept = MultiModalEventDataset(src, sources, split='all', min_points=1)
    assert 0 < len(kept) <= n_all


def test_event_identity_shape(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=6)
    ds = MultiModalEventDataset(_SRC, [dict(root=root, label=0, config_id=5)])
    cid, file_index, sei = ds.event_identity(0)
    assert cid == 5
    assert isinstance(file_index, int) and isinstance(sei, int)

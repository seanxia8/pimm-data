"""MultiModalEventDataset: source mixture + deterministic hash holdout.

Wraps JAXTPCDataset per source. The headline property: the holdout is keyed on
(config_id, source_event_idx), so it is reproducible and invariant to how
events are sharded — the fix for the positional-permutation flaw (SR5/D26).
"""
import os

import numpy as np
import pytest

from pimm_data.testing import make_jaxtpc_sample
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


def test_holdout_invariant_to_sharding(tmp_path):
    """Same source_event_idx set, different shard layout → same split.

    Root A: 10 events in 1 file (sei 0..9). Root B: 10 events in 2 files of 5
    (sei 0..4, 5..9 → 0..9). Same config_id + seed ⇒ identical train split by
    source_event_idx. A positional permutation would NOT have this property.
    """
    a = make_jaxtpc_sample(str(tmp_path / 'a'), n_events=10, n_files=1)
    b = make_jaxtpc_sample(str(tmp_path / 'b'), n_events=5, n_files=2)
    ho = dict(seed=1, fractions=(0.6, 0.2, 0.2))
    da = MultiModalEventDataset(
        _SRC, [dict(root=a, label=0, config_id=0)], split='train', holdout=ho)
    db = MultiModalEventDataset(
        _SRC, [dict(root=b, label=0, config_id=0)], split='train', holdout=ho)
    # compare by source_event_idx (the stable id), ignoring root/local idx.
    assert {sei for _, sei in _identity_set(da)} == \
           {sei for _, sei in _identity_set(db)}


def test_n_per_config_holdout(tmp_path):
    srcs = _two_sources(tmp_path, n_events=20)
    ho = dict(seed=0, n_per_config=4)
    tr = MultiModalEventDataset(_SRC, srcs, split='train', holdout=ho)
    ho_ds = MultiModalEventDataset(_SRC, srcs, split='val', holdout=ho)
    # 4 held out per source × 2 sources = 8 held out; 40 - 8 = 32 train.
    assert len(ho_ds) == 8
    assert len(tr) == 32
    assert _identity_set(tr) & _identity_set(ho_ds) == set()


def test_event_identity_shape(tmp_path):
    root = make_jaxtpc_sample(str(tmp_path), n_events=6)
    ds = MultiModalEventDataset(_SRC, [dict(root=root, label=0, config_id=5)])
    cid, sei = ds.event_identity(0)
    assert cid == 5
    assert isinstance(sei, int)

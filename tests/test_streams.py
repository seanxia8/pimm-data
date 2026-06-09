"""Streams — named transform-views of one event (SSL / multi-scale / per-branch).

A stream is the same event processed differently (independent randomness),
namespaced under its name; collate packs it with no special handling.
"""
import numpy as np
import pytest
import torch

from pimm_data.transform import TRANSFORMS
from pimm_data import collate_fn


def _event():
    base = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], np.float32)
    return {'step': {'coord': base.copy(), 'energy': np.ones((3, 1), np.float32)},
            'name': 'evt0', 'split': 'train'}


def _two_view_pipe(shared_collect_only=False):
    # transforms hardcoding 'coord' must be scoped via ApplyToModality even inside
    # a stream (the stream pipeline runs on the whole event dict).
    aug = [] if shared_collect_only else [dict(type='ApplyToModality', modality='step',
              transforms=[dict(type='RandomRotate', angle=[-1, 1], axis='z', p=1.0)])]
    one = aug + [dict(type='Collect', modality='step', keys=('coord',), feat_keys=('coord', 'energy'))]
    return TRANSFORMS.build(dict(type='Streams', streams={'global': list(one), 'local': list(one)}))


def test_streams_namespaces_under_stream_name():
    np.random.seed(0)
    out = _two_view_pipe()( _event())
    assert set(out) == {'global', 'local', 'name', 'split'}
    for s in ('global', 'local'):
        assert set(out[s]) >= {'coord', 'feat', 'offset'}
        assert isinstance(out[s]['coord'], torch.Tensor)


def test_streams_views_are_independent():
    """Two views get INDEPENDENT augmentation (the SSL point) — they differ."""
    np.random.seed(0)
    out = _two_view_pipe()(_event())
    assert not torch.allclose(out['global']['coord'], out['local']['coord'])


def test_streams_identity_at_top_level_only():
    np.random.seed(0)
    out = _two_view_pipe()(_event())
    assert out['name'] == 'evt0' and out['split'] == 'train'
    assert 'name' not in out['global'] and 'split' not in out['local']


def test_streams_collate_packs_nesting():
    np.random.seed(0)
    pipe = _two_view_pipe(shared_collect_only=True)   # deterministic (no aug)
    batch = collate_fn([pipe(_event()), pipe(_event())])
    assert set(batch) == {'global', 'local', 'name', 'split'}
    assert batch['global']['offset'].shape == (2,)        # per-stream offset, B=2
    assert batch['global']['coord'].shape[0] == 6         # 3+3 concatenated
    assert batch['name'] == ['evt0', 'evt0']


def test_streams_rejects_empty_and_reserved_names():
    with pytest.raises(ValueError, match="at least one"):
        TRANSFORMS.build(dict(type='Streams', streams={}))
    with pytest.raises(ValueError, match="reserved"):
        TRANSFORMS.build(dict(type='Streams', streams={'name': []}))

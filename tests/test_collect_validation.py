"""Collect build-time validation (REDESIGN §9) — fail at construction, not epoch 1."""
import pytest

from pimm_data.transform import Collect


def test_namespaced_spec_requires_keys():
    with pytest.raises(ValueError, match="needs 'keys='"):
        Collect(parts={'step': dict(feat_keys=('coord',))})   # no keys=


def test_roles_for_non_collected_key_rejected():
    with pytest.raises(ValueError, match="non-collected keys"):
        Collect(parts={'step': dict(keys=('coord',),
                                          roles={'edge_index': ('edge', 'self')})})  # not in keys


def test_valid_namespaced_spec_ok():
    Collect(parts={'step': dict(keys=('coord', 'edge_index'),
                                     roles={'edge_index': ('edge', 'self')})})


def test_rejects_both_forms():
    with pytest.raises(AssertionError):
        Collect(keys=['coord'], parts={'step': dict(keys=['coord'])})


def test_deprecated_aliases_still_work():
    """modalities=/modality= are deprecated aliases for parts=/part=."""
    import numpy as np
    # multi: modalities= aliases parts=
    a = Collect(parts={'step': dict(keys=('coord',))})
    b = Collect(modalities={'step': dict(keys=('coord',))})
    assert a.parts_spec == b.parts_spec
    # single: modality= aliases part=, and produces the same bare output
    data = {'step': {'coord': np.zeros((3, 3), np.float32)}, 'name': 'e', 'split': 't'}
    out_new = Collect(part='step', keys=['coord'])(dict(step=data['step'], name='e', split='t'))
    out_old = Collect(modality='step', keys=['coord'])(dict(step=data['step'], name='e', split='t'))
    assert set(out_new) == set(out_old) == {'coord', 'offset', 'name', 'split'}

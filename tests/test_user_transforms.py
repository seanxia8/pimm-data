"""Uniform user-transform authoring.

A user writes ONE dict->dict callable and places it in the SAME Compose, pre- or
post-collate — there is no separate batch-transform runner. These tests lock that
Compose runs bare callables and the dense ops (scope='sample'), and that the
scope='batch' fence still catches a genuine cross-sample transform pre-collate.
"""
import numpy as np
import pytest
import torch

from pimm_data.transform import Compose
from pimm_data.batch_transforms import (BatchDensify, BatchAddIntrinsicNoise,
                                        BatchDigitize, ToDevice)


def test_precollate_bare_callable():
    """Compose runs a bare user callable fn(data)->data (pre-collate, per-event)."""
    seen = {}

    def tag_it(d):
        seen['ran'] = True
        d['marker'] = 1
        return d

    out = Compose([tag_it])({'coord': np.zeros((2, 3), np.float32)})
    assert seen.get('ran') and out['marker'] == 1


def test_postcollate_bare_callable_in_compose():
    """A bare user fn(batch)->batch runs post-collate via the ORDINARY Compose —
    no runner, no seeds plumbing. ToDevice is just another transform in the list."""
    def gpu_scale(batch):
        batch['feat'] = batch['feat'] * 2
        return batch

    batch = {'feat': torch.ones(4, 2), 'offset': torch.tensor([4]), 'name': ['e0']}
    out = Compose([dict(type='ToDevice', device='cpu'), gpu_scale])(batch)
    assert out['feat'].device == torch.device('cpu')
    assert torch.equal(out['feat'], torch.full((4, 2), 2.0))


def test_dense_ops_are_scope_sample_and_compose():
    """densify/noise/digitize are scope='sample' (per-single) -> run in a plain
    Compose alongside ToDevice (the fence does NOT block them)."""
    for cls in (ToDevice, BatchDensify, BatchAddIntrinsicNoise, BatchDigitize):
        assert getattr(cls, 'scope', None) == 'sample'
    Compose([dict(type='ToDevice', device='cpu'),
             dict(type='BatchDensify', geom={}, modality='sensor')])   # no fence error


def test_compose_rejects_genuine_scope_batch():
    """A genuine cross-sample (scope='batch') transform is still fenced pre-collate."""
    class CrossSample:
        scope = 'batch'
        def __call__(self, b):
            return b

    with pytest.raises(ValueError, match="scope='batch'"):
        Compose([CrossSample()])


# --- densify output key is neutral + modality-agnostic (no sensor assumption) ---

def _tiny_sensor_batch():
    # 2 events, one plane (gid 0), a 4x5 grid; event0 has 2 hits, event1 has 1.
    return dict(
        wire=torch.tensor([0, 3, 1]), time=torch.tensor([1, 4, 2]),
        value=torch.tensor([9., 7., 5.]), plane_gid=torch.tensor([0, 0, 0]),
        offset=torch.tensor([2, 3]), name=['a', 'b'])


def test_densify_default_key_is_neutral_dense():
    """BatchDensify writes the neutral 'dense' key — NO 'sensor_dense', and it works
    for ANY modality (densify isn't sensor-specific). Bare batch -> batch['dense']."""
    from pimm_data.batch_transforms import BatchDensify
    geom = {0: {'n_wires': 4, 'n_ticks': 5}}
    out = BatchDensify(geom)(_tiny_sensor_batch())          # bare, no dense_key=
    assert 'dense' in out and 'sensor_dense' not in out
    assert out['dense'][0].shape == (2, 4, 5)


def _flat_sensor_batch(**override):
    b = _tiny_sensor_batch()
    b.update(override)
    return {f'sensor_{k}': v for k, v in b.items() if k != 'name'}


def test_densify_namespaced_default_lands_in_flat_dense():
    """REDESIGN: modality='sensor' reads FLAT sensor_* keys and writes sensor_dense
    (flat-prefixed), not a nested batch['sensor']['dense']."""
    from pimm_data.batch_transforms import BatchDensify
    geom = {0: {'n_wires': 4, 'n_ticks': 5}}
    out = BatchDensify(geom, modality='sensor')(_flat_sensor_batch())   # no dense_key=
    assert 'sensor_dense' in out and out['sensor_dense'][0].shape == (2, 4, 5)
    assert 'dense' not in out                               # not bare


def test_densify_coord_mutation_coupling_is_loud():
    """The #1 dense coupling: a coord-mutating transform desyncs the COO from
    offset. BatchDensify names the modality and the fix — clearer than the
    generic dense_ops guard."""
    from pimm_data.batch_transforms import BatchDensify
    geom = {0: {'n_wires': 4, 'n_ticks': 5}}
    batch = _flat_sensor_batch(offset=torch.tensor([1, 2]))   # total 2 != 3 COO rows
    with pytest.raises(ValueError, match=r"BatchDensify\('sensor'\).*coord-mutating"):
        BatchDensify(geom, modality='sensor')(batch)

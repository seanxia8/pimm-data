"""Uniform user-transform authoring across the collate pivot.

The point of the restructure: a user writes ONE dict->dict callable and the
harness places it — pre-collate (per-event, CPU) via Compose, or post-collate
(per-batch, on device) via apply_batch_transforms — with no registration and no
two-variant burden. These tests lock that both ends accept bare callables and
that the scope='batch' fence catches a misplacement.
"""
import numpy as np
import pytest
import torch

from pimm_data.transform import Compose
from pimm_data.batch_transforms import (apply_batch_transforms, build_batch_transforms,
                                        BatchDensify, ToDevice)


def test_precollate_bare_callable():
    """Compose runs a bare user callable fn(data)->data (pre-collate, per-event)."""
    seen = {}

    def tag_it(d):
        seen['ran'] = True
        d['marker'] = 1
        return d

    out = Compose([tag_it])({'coord': np.zeros((2, 3), np.float32)})
    assert seen.get('ran') and out['marker'] == 1


def test_postcollate_bare_callable():
    """apply_batch_transforms runs a bare user callable fn(batch)->batch as a
    post-collate stage — same authoring as pre-collate, no seeds= plumbing."""
    def gpu_scale(batch):                       # a user GPU op; no seeds kwarg
        batch['feat'] = batch['feat'] * 2
        return batch

    batch = {'feat': torch.ones(4, 2), 'offset': torch.tensor([4]), 'name': ['e0']}
    out = apply_batch_transforms(batch, [gpu_scale], device='cpu')
    assert torch.equal(out['feat'], torch.full((4, 2), 2.0))


def test_postcollate_seeded_stage_still_gets_seeds():
    """A stage that DOES take seeds= still receives them (protocol unchanged)."""
    captured = {}

    def seeded(batch, *, seeds):
        captured['seeds'] = list(seeds)
        return batch

    batch = {'offset': torch.tensor([2, 5]), 'name': ['a', 'b']}
    apply_batch_transforms(batch, [seeded], device='cpu', base_seed=7)
    assert len(captured['seeds']) == 2 and all(isinstance(s, int) for s in captured['seeds'])


def test_mixed_user_and_library_stages():
    """User bare callable and a library scope='batch' stage compose in one list."""
    calls = []
    out = apply_batch_transforms(
        {'feat': torch.ones(1, 1), 'offset': torch.tensor([1]), 'name': ['e0']},
        [lambda b: (calls.append('user'), b)[1],
         lambda b, *, seeds: (calls.append('seeded'), b)[1]],
        device='cpu')
    assert calls == ['user', 'seeded']


def test_compose_rejects_scope_batch():
    """A scope='batch' transform placed pre-collate is caught at build time."""
    stage = BatchDensify({}, modality='sensor')
    assert getattr(stage, 'scope', None) == 'batch'
    with pytest.raises(ValueError, match="scope='batch'"):
        Compose([stage])


# --- general batch-transform builder + device-as-transform ------------------

def test_build_batch_transforms_from_config_and_callables():
    """The post-collate analog of Compose: builds registry dicts AND bare callables
    into one stage list — no bespoke per-purpose builder needed."""
    stages = build_batch_transforms([
        dict(type='ToDevice', device='cpu'),
        lambda b: b,
    ])
    assert isinstance(stages[0], ToDevice) and stages[0].device == 'cpu'
    assert callable(stages[1])


def test_todevice_as_transform_equiv_to_device_arg():
    """ToDevice as the first stage == passing device= to the runner (same result)."""
    def mk():
        return {'feat': torch.ones(3, 2), 'offset': torch.tensor([3]), 'name': ['e0']}

    via_arg   = apply_batch_transforms(mk(), [lambda b: b], device='cpu')
    via_stage = apply_batch_transforms(
        mk(), build_batch_transforms([dict(type='ToDevice', device='cpu'),
                                      lambda b: b]))   # no device= argument
    assert via_stage['feat'].device == via_arg['feat'].device
    assert torch.equal(via_stage['feat'], via_arg['feat'])


def test_build_batch_transforms_warns_on_scope_sample():
    """A scope='sample' transform in a batch list is built but warned (it belongs
    in the pre-collate Compose)."""
    class PerEvent:
        scope = 'sample'
        def __call__(self, b):
            return b

    with pytest.warns(RuntimeWarning, match="scope='sample'"):
        build_batch_transforms([PerEvent()])


def test_compose_rejects_todevice():
    """ToDevice is scope='batch' → the pre-collate fence rejects it too."""
    with pytest.raises(ValueError, match="scope='batch'"):
        Compose([dict(type='ToDevice', device='cpu')])


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


def test_densify_namespaced_default_lands_in_modality_dense():
    """modality='sensor' with NO dense_key= now lands at batch['sensor']['dense']
    (the de-footgunned default), not batch['sensor']['sensor_dense']."""
    from pimm_data.batch_transforms import BatchDensify
    geom = {0: {'n_wires': 4, 'n_ticks': 5}}
    batch = {'sensor': _tiny_sensor_batch()}
    out = BatchDensify(geom, modality='sensor')(batch)      # no dense_key=
    assert 'dense' in out['sensor'] and 'sensor_dense' not in out['sensor']

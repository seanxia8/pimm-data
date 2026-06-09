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
from pimm_data.batch_transforms import apply_batch_transforms, BatchDensify


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

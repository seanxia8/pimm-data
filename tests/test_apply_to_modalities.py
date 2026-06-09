"""ApplyToModalities — one sub-pipeline over several modalities with a shared
random draw (consistent, co-registered geometric augmentation)."""
import numpy as np
import pytest

from pimm_data.transform import TRANSFORMS


def _two_identical():
    base = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], np.float32)
    return {'step': {'coord': base.copy()}, 'sensor': {'coord': base.copy()}}


def _rot_block(modalities, shared):
    return TRANSFORMS.build(dict(
        type='ApplyToModalities', modalities=modalities, shared=shared,
        transforms=[dict(type='RandomRotate', angle=[-1, 1], axis='z', p=1.0)]))


def test_shared_draw_keeps_modalities_aligned():
    """shared=True: identical inputs get the SAME rotation -> still equal."""
    np.random.seed(1)
    out = _rot_block(['step', 'sensor'], shared=True)(_two_identical())
    assert np.allclose(out['step']['coord'], out['sensor']['coord'])


def test_independent_draw_diverges():
    """shared=False: independent draws -> the two modalities rotate differently."""
    np.random.seed(1)
    out = _rot_block(['step', 'sensor'], shared=False)(_two_identical())
    assert not np.allclose(out['step']['coord'], out['sensor']['coord'])


def test_shared_still_actually_transforms():
    """Sanity: shared mode isn't a no-op — coords DID rotate away from input."""
    np.random.seed(2)
    d = _two_identical()
    before = d['step']['coord'].copy()
    out = _rot_block(['step', 'sensor'], shared=True)(d)
    assert not np.allclose(out['step']['coord'], before)


def test_missing_modality_skipped_unless_required():
    np.random.seed(0)
    d = {'step': {'coord': np.ones((2, 3), np.float32)}}      # no 'sensor'
    out = _rot_block(['step', 'sensor'], shared=True)(d)       # sensor absent -> skipped
    assert 'sensor' not in out and 'step' in out
    with pytest.raises(KeyError, match="missing"):
        TRANSFORMS.build(dict(type='ApplyToModalities', modalities=['step', 'sensor'],
                              required=True, transforms=[dict(type='RandomFlip')]))(d)


def test_rng_advances_after_shared_block():
    """After a shared block the global RNG advanced once (not reset), so a
    following draw is not frozen to the pre-block value."""
    np.random.seed(3)
    pre = np.random.get_state()[1][0]
    _rot_block(['step', 'sensor'], shared=True)(_two_identical())
    post = np.random.get_state()[1][0]
    assert pre != post

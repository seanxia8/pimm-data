"""Task-matrix combination tests for JAXTPCDataset nested output.

Drives the schema defined in pimm-data's refactor: every combination
produces a nested dict {'step': {...}, 'sensor': {...}, 'hits': {...},
'labl': {...}, 'bridges': {...}, 'name', 'split'}. Missing modalities
have no top-level key. No bare 'coord', no flat namespaced aliases.

Rows cross-reference the modality matrix in README §Modality combinations.
"""

import numpy as np
import pytest

from pimm_data import JAXTPCDataset


def make_ds(root, modalities, **kw):
    defaults = dict(data_root=root, split='', dataset_name='sim',
                    modalities=modalities)
    defaults.update(kw)
    return JAXTPCDataset(**defaults)


def _assert_point_cloud(sub, expect_dim, labeled=False,
                        extra_required=()):
    """Every point-cloud sub-dict has coord/energy/plane_id consistent."""
    assert 'coord' in sub, f"missing coord in sub-dict keys: {list(sub)}"
    assert sub['coord'].ndim == 2
    assert sub['coord'].shape[1] == expect_dim, \
        f"coord dim {sub['coord'].shape[1]} != {expect_dim}"
    n = sub['coord'].shape[0]
    assert sub['energy'].shape == (n, 1)
    if 'plane_id' in sub:
        assert sub['plane_id'].shape == (n, 1)
    if labeled:
        assert 'segment' in sub and sub['segment'].shape == (n,)
        assert 'instance' in sub and sub['instance'].shape == (n,)
    for k in extra_required:
        assert k in sub, f"missing {k} in {list(sub)}"


def _top_keys(d):
    return set(d.keys()) - {'name', 'split'}


# ---------- Row 1 / 4: SSL raw sensor / real-data inference ----------

def test_row1_ssl_raw_sensor(jaxtpc_data_root):
    """sensor only — sensor cloud, no bridges, no labl, no labels."""
    ds = make_ds(jaxtpc_data_root, modalities=('sensor',))
    d = ds.get_data(0)
    assert _top_keys(d) == {'sensor'}
    _assert_point_cloud(d['sensor'], expect_dim=2)
    assert 'segment' not in d['sensor']
    assert 'raw' in d['sensor']
    # raw is nested per plane
    some_plane = next(iter(d['sensor']['raw']))
    assert {'wire', 'time', 'value'} <= set(d['sensor']['raw'][some_plane])


# ---------- Row 2: SSL on clean inst ----------

def test_row2_ssl_hits(jaxtpc_data_root):
    """hits only — hits cloud + bridges, no labels."""
    ds = make_ds(jaxtpc_data_root, modalities=('hits',))
    d = ds.get_data(0)
    assert _top_keys(d) == {'hits', 'bridges'}
    _assert_point_cloud(d['hits'], expect_dim=2, extra_required=('instance',))
    assert 'segment' not in d['hits']
    assert 'raw' in d['hits']
    # bridges populated (per-volume)
    assert any(k.startswith('group_to_track_v') for k in d['bridges'])
    assert any(k.startswith('deposit_to_group_v') for k in d['bridges'])


# ---------- Row 3 / 11: SSL on seg / per-deposit regression ----------

def test_row3_ssl_step(jaxtpc_data_root):
    """step only — 3D cloud with physics, no labels."""
    ds = make_ds(jaxtpc_data_root, modalities=('step',))
    d = ds.get_data(0)
    assert _top_keys(d) == {'step'}
    _assert_point_cloud(d['step'], expect_dim=3)
    assert 'segment' not in d['step']
    assert 'instance' not in d['step']


def test_row11_physics_regression(jaxtpc_data_root):
    """seg with include_physics — charge/photons/dx/theta etc present."""
    ds = make_ds(jaxtpc_data_root, modalities=('step',),
                 include_physics=True)
    d = ds.get_data(0)
    for key in ('dx', 'theta', 'phi', 't0_us', 'charge', 'photons'):
        if key in d['step']:
            assert d['step'][key].shape == (d['step']['coord'].shape[0], 1), \
                f"{key} shape unexpected"


# ---------- Row 5: sensor + inst (denoising) ----------

def test_row5_denoising(jaxtpc_data_root):
    """sensor + hits — both clouds in parallel, bridges present, no labels."""
    ds = make_ds(jaxtpc_data_root, modalities=('sensor', 'hits'))
    d = ds.get_data(0)
    assert _top_keys(d) == {'sensor', 'hits', 'bridges'}
    _assert_point_cloud(d['sensor'], expect_dim=2)
    _assert_point_cloud(d['hits'], expect_dim=2, extra_required=('instance',))
    assert 'segment' not in d['sensor']
    assert 'segment' not in d['hits']


# ---------- Row 6 / 10: sensor + seg ----------

def test_row6_sensor_step(jaxtpc_data_root):
    """sensor + step — no direct bridge today but both clouds must load."""
    ds = make_ds(jaxtpc_data_root, modalities=('sensor', 'step'))
    d = ds.get_data(0)
    assert _top_keys(d) == {'sensor', 'step'}
    _assert_point_cloud(d['sensor'], expect_dim=2)
    _assert_point_cloud(d['step'], expect_dim=3)


# ---------- Row 7: supervised on inst ----------

def test_row7_supervised_hits(jaxtpc_data_root):
    """hits + labl — hits cloud with segment/instance, labl tables."""
    ds = make_ds(jaxtpc_data_root, modalities=('hits', 'labl'),
                 label_key='pdg')
    d = ds.get_data(0)
    assert _top_keys(d) == {'hits', 'labl', 'bridges'}
    _assert_point_cloud(d['hits'], expect_dim=2, labeled=True)
    # labl is per-volume
    assert all(vk.startswith('v') for vk in d['labl'])
    some_vol = next(iter(d['labl']))
    assert 'track_ids' in d['labl'][some_vol]


# ---------- Row 8: supervised on seg ----------

def test_row8_supervised_step(jaxtpc_data_root):
    """step + labl — 3D with segment/instance from labl FK."""
    ds = make_ds(jaxtpc_data_root, modalities=('step', 'labl'),
                 label_key='pdg')
    d = ds.get_data(0)
    assert _top_keys(d) == {'step', 'labl'}
    _assert_point_cloud(d['step'], expect_dim=3, labeled=True)
    some_vol = next(iter(d['labl']))
    assert 'deposit_to_track' in d['labl'][some_vol]


# ---------- Row 9: 2D end-to-end ----------

def test_row9_2d_end_to_end(jaxtpc_data_root):
    """sensor + hits + labl — denoising target with supervision on hits."""
    ds = make_ds(jaxtpc_data_root,
                 modalities=('sensor', 'hits', 'labl'),
                 label_key='pdg')
    d = ds.get_data(0)
    assert _top_keys(d) == {'sensor', 'hits', 'labl', 'bridges'}
    _assert_point_cloud(d['sensor'], expect_dim=2)
    _assert_point_cloud(d['hits'], expect_dim=2, labeled=True)
    assert 'segment' not in d['sensor']  # sensor never carries labels


# ---------- Row 10: 3D end-to-end ----------

def test_row10_3d_end_to_end(jaxtpc_data_root):
    """sensor + step + labl — sensor to 3D supervised reconstruction."""
    ds = make_ds(jaxtpc_data_root,
                 modalities=('sensor', 'step', 'labl'),
                 label_key='pdg')
    d = ds.get_data(0)
    assert _top_keys(d) == {'sensor', 'step', 'labl'}
    _assert_point_cloud(d['sensor'], expect_dim=2)
    _assert_point_cloud(d['step'], expect_dim=3, labeled=True)


# ---------- Row 13: joint multi-task (all four) ----------

def test_row13_joint_multitask(jaxtpc_data_root):
    """All four modalities — every cloud labeled where possible."""
    ds = make_ds(jaxtpc_data_root,
                 modalities=('step', 'sensor', 'hits', 'labl'),
                 label_key='pdg')
    d = ds.get_data(0)
    assert _top_keys(d) == {'step', 'sensor', 'hits', 'labl', 'bridges'}
    _assert_point_cloud(d['step'], expect_dim=3, labeled=True)
    _assert_point_cloud(d['sensor'], expect_dim=2)
    _assert_point_cloud(d['hits'], expect_dim=2, labeled=True)


# ---------- Invalid combinations (see README §Modality combinations) ----------

def test_invalid_sensor_plus_labl_only(jaxtpc_data_root):
    """sensor+labl alone must raise — no bridge to attach labels."""
    with pytest.raises(ValueError, match=r"hits|bridging|sensor"):
        make_ds(jaxtpc_data_root, modalities=('sensor', 'labl'))


def test_invalid_labl_only(jaxtpc_data_root):
    """labl alone must raise — dimension table has nothing to join."""
    with pytest.raises(ValueError, match=r"dimension|labl"):
        make_ds(jaxtpc_data_root, modalities=('labl',))


# ---------- Cross-structure invariants ----------

def test_no_bare_coord_anywhere(jaxtpc_data_root):
    """No top-level bare 'coord', 'energy', 'segment' — all namespaced."""
    for mods in [('step',), ('sensor',), ('hits',),
                 ('step', 'labl'), ('sensor', 'hits'),
                 ('hits', 'labl'),
                 ('step', 'sensor', 'hits', 'labl')]:
        ds = make_ds(jaxtpc_data_root, modalities=mods, label_key='pdg')
        d = ds.get_data(0)
        forbidden = {'coord', 'energy', 'segment', 'instance',
                     'plane_id', 'volume_id'}
        bare = forbidden & set(d.keys())
        assert not bare, f"bare keys {bare} leaked in modalities={mods}"


def test_bridges_present_iff_inst(jaxtpc_data_root):
    """'bridges' top-level key present exactly when 'hits' is loaded."""
    cases = [
        (('step',), False),
        (('sensor',), False),
        (('hits',), True),
        (('step', 'labl'), False),
        (('sensor', 'hits'), True),
        (('hits', 'labl'), True),
    ]
    for mods, expect in cases:
        ds = make_ds(jaxtpc_data_root, modalities=mods, label_key='pdg')
        d = ds.get_data(0)
        has_bridges = 'bridges' in d
        assert has_bridges == expect, \
            f"bridges={has_bridges} for {mods}; expected {expect}"


def test_labl_is_per_volume(jaxtpc_data_root):
    """labl sub-dict is keyed per-volume (v0, v1, ...) with column sub-dicts."""
    ds = make_ds(jaxtpc_data_root, modalities=('step', 'labl'),
                 label_key='pdg')
    d = ds.get_data(0)
    for vk, vdata in d['labl'].items():
        assert vk.startswith('v'), f"unexpected labl key {vk}"
        assert isinstance(vdata, dict)
        assert 'track_ids' in vdata


def test_sensor_and_hits_raw_are_nested(jaxtpc_data_root):
    """sensor['raw'] and hits['raw'] are nested by plane, not flat dotted."""
    ds = make_ds(jaxtpc_data_root,
                 modalities=('sensor', 'hits'),
                 label_key='pdg')
    d = ds.get_data(0)
    # No flat 'sensor.volume_X_Y.wire' at top level
    dotted = [k for k in d if k.startswith(('sensor.', 'hits.'))]
    assert not dotted, f"dotted keys leaked: {dotted[:5]}"
    for raw in (d['sensor']['raw'], d['hits']['raw']):
        for plane, cols in raw.items():
            assert isinstance(cols, dict)
            assert 'wire' in cols and 'time' in cols


# ---------- Enforce the README §Modality combinations matrix ----------
#
# Every row of the matrix is expressed as (combo, expected truth-flags). This
# test fails if any future dataset change drifts from the documented matrix.

_MATRIX = [
    # combo,                                    step sensor hits labl bridges step_seg step_inst hits_seg
    (('step',),                                (True, False, False, False, False, False, False, False)),
    (('sensor',),                              (False, True, False, False, False, False, False, False)),
    (('hits',),                                (False, False, True, False, True, False, False, False)),
    (('step', 'sensor'),                       (True, True, False, False, False, False, False, False)),
    (('step', 'hits'),                         (True, False, True, False, True, False, False, False)),
    (('sensor', 'hits'),                       (False, True, True, False, True, False, False, False)),
    (('step', 'labl'),                         (True, False, False, True, False, True, True, False)),
    (('hits', 'labl'),                         (False, False, True, True, True, False, False, True)),
    (('step', 'sensor', 'hits'),               (True, True, True, False, True, False, False, False)),
    (('step', 'sensor', 'labl'),               (True, True, False, True, False, True, True, False)),
    (('step', 'hits', 'labl'),                 (True, False, True, True, True, True, True, True)),
    (('sensor', 'hits', 'labl'),               (False, True, True, True, True, False, False, True)),
    (('step', 'sensor', 'hits', 'labl'),       (True, True, True, True, True, True, True, True)),
]


@pytest.mark.parametrize("combo,flags", _MATRIX, ids=lambda x: str(x))
def test_combination_matrix(jaxtpc_data_root, combo, flags):
    """Every documented combo matches live output."""
    ds = make_ds(jaxtpc_data_root, modalities=combo, label_key='pdg')
    d = ds.get_data(0)
    step, sensor, hits, labl, bridges, step_seg, step_inst, hits_seg = flags
    assert ('step' in d) == step
    assert ('sensor' in d) == sensor
    assert ('hits' in d) == hits
    assert ('labl' in d) == labl
    assert ('bridges' in d) == bridges
    has_step_seg = 'step' in d and 'segment' in d['step']
    has_step_inst = 'step' in d and 'instance' in d['step']
    has_hits_seg = 'hits' in d and 'segment' in d['hits']
    assert has_step_seg == step_seg
    assert has_step_inst == step_inst
    assert has_hits_seg == hits_seg

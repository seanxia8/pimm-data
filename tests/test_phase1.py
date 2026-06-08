"""Phase 1 — labels (labl -> labels=), G1 seeding, G2 name pass-through."""
import hashlib

import numpy as np

from pimm_data import JAXTPCDataset
from pimm_data.transform import Collect


def _seg_fp(a):
    a = np.ascontiguousarray(a)
    return (tuple(a.shape), str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest()[:16])


def _jds(root, modalities, **kw):
    return JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                         modalities=modalities, min_deposits=0, max_len=2, **kw)


def test_labels_param_matches_legacy_labl_modality(jaxtpc_data_root):
    """modalities=('step',), labels='pdg' decorates identically to the legacy
    modalities=('step','labl'), label_key='pdg' — labl is now a label source."""
    new = _jds(jaxtpc_data_root, ('step',), labels='pdg').get_data(0)['step']
    old = _jds(jaxtpc_data_root, ('step', 'labl'), label_key='pdg').get_data(0)['step']
    assert _seg_fp(new['segment']) == _seg_fp(old['segment'])
    assert _seg_fp(new['instance']) == _seg_fp(old['instance'])


def test_labels_requires_decoratable_modality(jaxtpc_data_root):
    import pytest
    with pytest.raises(ValueError, match="decoratable"):
        _jds(jaxtpc_data_root, ('sensor',), labels='pdg')


def test_lucid_labels_param_matches_legacy(lucid_data_root):
    """LUCiD: modalities=('hits',), labels=True decorates identically to the
    legacy modalities=('hits','labl')."""
    from pimm_data import LUCiDDataset

    def ld(mods, **kw):
        return LUCiDDataset(data_root=lucid_data_root, split='', dataset_name='wc',
                            modalities=mods, max_len=2, **kw)

    new = ld(('hits',), labels=True).get_data(0)['hits']
    old = ld(('hits', 'labl')).get_data(0)['hits']
    assert set(new) == set(old)
    for k in ('segment', 'instance'):
        if k in new:
            assert _seg_fp(new[k]) == _seg_fp(old[k])


# --- Phase 2: namespaced multi-modality Collect ---------------------------

def test_namespaced_multimodality_collect(jaxtpc_data_root):
    """Collect(modalities={...}) -> {step:{...}, sensor:{...}, name, split} with
    each modality self-contained, its OWN offset, and junk (raw/coord) dropped."""
    from pimm_data import collate_fn
    ds = JAXTPCDataset(
        data_root=jaxtpc_data_root, split='', dataset_name='sim',
        modalities=('step', 'sensor'), labels='pdg', min_deposits=0, max_len=2,
        transform=[dict(type='Collect', modalities={
            'step':   dict(keys=('coord', 'segment'), feat_keys=('coord', 'energy')),
            'sensor': dict(keys=('wire', 'time', 'value', 'plane_gid')),
        })])
    batch = collate_fn([ds[0], ds[1]])

    assert set(batch) == {'step', 'sensor', 'name', 'split'}          # namespaced
    assert set(batch['step']) >= {'coord', 'segment', 'feat', 'offset'}
    assert set(batch['sensor']) == {'wire', 'time', 'value', 'plane_gid', 'offset'}
    assert 'raw' not in batch['sensor'] and 'coord' not in batch['sensor']  # junk gone
    # each modality has its OWN cumulative offset (B,), matching its row count
    assert batch['step']['offset'].shape == (2,)
    assert batch['sensor']['offset'].shape == (2,)
    assert int(batch['step']['offset'][-1]) == batch['step']['coord'].shape[0]
    assert int(batch['sensor']['offset'][-1]) == batch['sensor']['wire'].shape[0]
    assert batch['step']['coord'].shape[1] == 3 and batch['sensor']['wire'].ndim == 1


def test_collect_rejects_both_forms():
    import pytest
    from pimm_data.transform import Collect
    with pytest.raises(AssertionError):
        Collect(keys=['coord'], modalities={'step': dict(keys=['coord'])})


def test_sensor_consumed_sparse_no_densify(jaxtpc_data_root):
    """No dense assumption: sensor is a sparse modality. A model may collect it
    as a 2D point cloud (coord/feat) with NO densify step — densify is opt-in."""
    from pimm_data import collate_fn
    ds = JAXTPCDataset(
        data_root=jaxtpc_data_root, split='', dataset_name='sim',
        modalities=('step', 'sensor'), labels='pdg', min_deposits=0, max_len=2,
        transform=[dict(type='Collect', modalities={
            'step':   dict(keys=('coord', 'segment'), feat_keys=('coord', 'energy')),
            'sensor': dict(keys=('coord',), feat_keys=('coord', 'energy')),
        })])
    batch = collate_fn([ds[0], ds[1]])
    assert batch['sensor']['coord'].shape[1] == 2          # 2D wire×time point cloud
    assert 'feat' in batch['sensor'] and 'offset' in batch['sensor']
    assert 'dense' not in batch['sensor']                  # nothing forced dense


# --- Phase 3: optional densify scoped to a namespaced modality ------------

def test_dense_stage_scoped_to_namespaced_modality(jaxtpc_data_root):
    """Densify/noise/digitize scoped to a modality → batch['sensor']['dense'] =
    {plane_gid:(B,W,T)}, born on the runner's device (CPU here, device-agnostic);
    other modalities untouched. Seeds come from top-level batch['name']."""
    import torch
    from pimm_data import (collate_fn, apply_batch_transforms,
                           build_sensor_gpu_stages)
    wl = {'U': (0.42, 4.63), 'V': (0.42, 4.63), 'Y': (2.33, 2.33)}
    ds = JAXTPCDataset(
        data_root=jaxtpc_data_root, split='', dataset_name='sim',
        modalities=('step', 'sensor'), labels='pdg', min_deposits=0, max_len=2,
        wire_lengths_per_plane=wl,
        transform=[dict(type='Collect', modalities={
            'step':   dict(keys=('coord', 'segment'), feat_keys=('coord', 'energy')),
            'sensor': dict(keys=('wire', 'time', 'value', 'plane_gid')),
        })])
    ds.get_data(0)                       # populate reader geometry
    geom = ds.plane_geometry()
    batch = collate_fn([ds[0], ds[1]])
    step_coord = batch['step']['coord'].clone()

    stages = build_sensor_gpu_stages(geom, modality='sensor', coherent=True,
                                     incoherent=False, digitize=True)
    out = apply_batch_transforms(batch, stages, device='cpu', base_seed=0, epoch=0)

    grids = out['sensor']['dense']                          # born under sensor
    assert isinstance(grids, dict) and len(grids) >= 1
    for g in grids.values():
        assert g.ndim == 3 and g.shape[0] == 2              # (B, W, T)
    assert 'dense' not in out['step'] and torch.equal(out['step']['coord'], step_coord)


def test_g2_bare_collect_passes_name_split():
    # modality=None (bare) Collect must still carry the identity keys.
    data = {'coord': np.zeros((3, 3), np.float32), 'name': 'evt0', 'split': 'train'}
    out = Collect(keys=['coord'])(data)
    assert out['name'] == 'evt0'
    assert out['split'] == 'train'
    assert tuple(out['offset'].tolist()) == (3,)


def test_g2_modality_collect_passes_name_split():
    data = {'step': {'coord': np.zeros((3, 3), np.float32)},
            'name': 'evt0', 'split': 'train'}
    out = Collect(modality='step', keys=['coord'])(data)
    assert out['name'] == 'evt0' and out['split'] == 'train'

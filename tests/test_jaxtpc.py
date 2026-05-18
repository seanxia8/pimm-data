"""Tests for JAXTPCDataset core loading logic (no transforms).

These complement ``test_jaxtpc_task_matrix.py``: where the matrix tests
cover every modality combination structurally (see README §Modality
combinations), these test orthogonal concerns — volume filtering,
label_key variants, DataLoader fork-safety, and name/split metadata.
"""

import numpy as np
import pytest

from pimm_data import JAXTPCDataset


def make_ds(jaxtpc_data_root, **kwargs):
    defaults = dict(data_root=jaxtpc_data_root, split='', dataset_name='sim')
    defaults.update(kwargs)
    return JAXTPCDataset(**defaults)


def test_volume_filter(jaxtpc_data_root):
    """volume=0 — only volume 0 data (fewer points than all volumes)."""
    ds_all = make_ds(jaxtpc_data_root, modalities=('sensor',))
    ds_v0 = make_ds(jaxtpc_data_root, modalities=('sensor',), volume=0)
    d_all = ds_all.get_data(0)
    d_v0 = ds_v0.get_data(0)
    assert d_v0['sensor']['coord'].shape[0] < d_all['sensor']['coord'].shape[0]


@pytest.mark.parametrize('label_key', ['pdg', 'cluster', 'interaction'])
def test_different_label_keys(jaxtpc_data_root, label_key):
    ds = make_ds(jaxtpc_data_root,
                 modalities=('edep', 'labl'), label_key=label_key)
    d = ds.get_data(0)
    assert len(np.unique(d['edep']['segment'])) > 1


def test_len_and_getitem(jaxtpc_data_root):
    ds = make_ds(jaxtpc_data_root, modalities=('edep',))
    assert len(ds) > 0
    sample = ds[0]
    assert isinstance(sample, dict)
    assert isinstance(sample['edep']['coord'], np.ndarray)


def test_name_and_split(jaxtpc_data_root):
    ds = make_ds(jaxtpc_data_root, modalities=('edep',))
    d = ds.get_data(0)
    assert 'name' in d
    assert 'split' in d


def test_dataloader_workers(jaxtpc_data_root):
    """Dataset is fork-safe via lazy h5py_worker_init()."""
    import torch
    ds = make_ds(jaxtpc_data_root, modalities=('edep', 'labl'),
                 label_key='pdg', min_deposits=0)
    if len(ds) < 2:
        pytest.skip("Need at least 2 events")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=lambda batch: batch)
    seen = 0
    for batch in loader:
        assert isinstance(batch[0], dict)
        assert 'edep' in batch[0]
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1


def test_physics_fields_present(jaxtpc_data_root):
    ds = make_ds(jaxtpc_data_root, modalities=('edep',), include_physics=True)
    d = ds.get_data(0)
    for key in ('dx', 'theta', 'phi'):
        if key in d['edep']:  # present iff h5 had the field
            assert d['edep'][key].shape[1] == 1


def test_empty_modalities_raises(jaxtpc_data_root):
    with pytest.raises(ValueError, match='empty'):
        make_ds(jaxtpc_data_root, modalities=())


def test_unknown_modality_raises(jaxtpc_data_root):
    with pytest.raises(ValueError, match='Unknown'):
        make_ds(jaxtpc_data_root, modalities=('edep', 'mystery'))

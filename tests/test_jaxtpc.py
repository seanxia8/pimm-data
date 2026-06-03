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
                 modalities=('step', 'labl'), label_key=label_key)
    d = ds.get_data(0)
    assert len(np.unique(d['step']['segment'])) > 1


def test_len_and_getitem(jaxtpc_data_root):
    ds = make_ds(jaxtpc_data_root, modalities=('step',))
    assert len(ds) > 0
    sample = ds[0]
    assert isinstance(sample, dict)
    assert isinstance(sample['step']['coord'], np.ndarray)


def test_name_and_split(jaxtpc_data_root):
    ds = make_ds(jaxtpc_data_root, modalities=('step',))
    d = ds.get_data(0)
    assert 'name' in d
    assert 'split' in d


def test_dataloader_workers(jaxtpc_data_root):
    """Dataset is fork-safe via lazy h5py_worker_init()."""
    import torch
    ds = make_ds(jaxtpc_data_root, modalities=('step', 'labl'),
                 label_key='pdg', min_deposits=0)
    if len(ds) < 2:
        pytest.skip("Need at least 2 events")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=lambda batch: batch)
    seen = 0
    for batch in loader:
        assert isinstance(batch[0], dict)
        assert 'step' in batch[0]
        seen += 1
        if seen >= 2:
            break
    assert seen >= 1


def test_physics_fields_present(jaxtpc_data_root):
    ds = make_ds(jaxtpc_data_root, modalities=('step',), include_physics=True)
    d = ds.get_data(0)
    for key in ('dx', 'theta', 'phi'):
        if key in d['step']:  # present iff h5 had the field
            assert d['step'][key].shape[1] == 1


def test_empty_modalities_raises(jaxtpc_data_root):
    with pytest.raises(ValueError, match='empty'):
        make_ds(jaxtpc_data_root, modalities=())


def test_unknown_modality_raises(jaxtpc_data_root):
    with pytest.raises(ValueError, match='Unknown'):
        make_ds(jaxtpc_data_root, modalities=('step', 'mystery'))


def test_volume_filter_prunes_bridges(tmp_path):
    """F13: under a volume= filter, bridges carry only that volume's group
    machinery — not the other volume's (whose points are never loaded)."""
    from pimm_data.testing import make_jaxtpc_sample
    root = make_jaxtpc_sample(str(tmp_path), n_events=2, n_volumes=2)
    kw = dict(data_root=root, split='', dataset_name='sim',
              modalities=('hits', 'labl'))
    b_all = JAXTPCDataset(**kw).get_data(0)['bridges']
    b_v0 = JAXTPCDataset(volume=0, **kw).get_data(0)['bridges']
    # all-volumes payload spans both volumes...
    assert any(k.endswith('_v1') for k in b_all)
    assert any(k.endswith('_v0') for k in b_all)
    # ...volume=0 payload keeps v0 tables and drops v1 (no orphan machinery).
    assert any(k.endswith('_v0') for k in b_v0)
    assert all(not k.endswith('_v1') for k in b_v0)


def test_min_deposits_cached_scan_matches_direct(tmp_path):
    """F16: the cached deposit-count scan produces the same min_deposits filter
    as a direct per-event n_actual read, and the scan is memoized."""
    import os
    import h5py
    from pimm_data.testing import make_jaxtpc_sample
    from pimm_data._shard_meta import read_deposit_counts, clear_cache
    clear_cache()
    root = make_jaxtpc_sample(str(tmp_path), n_events=4, n_volumes=2)
    step = os.path.join(root, 'step', 'sim_step_0000.h5')
    # zero event_001's deposits so min_deposits=1 must drop exactly it
    with h5py.File(step, 'r+') as f:
        for vk in [k for k in f['event_001'] if k.startswith('volume_')]:
            f['event_001'][vk].attrs['n_actual'] = 0
    clear_cache()
    ds = JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                       modalities=('step',), min_deposits=1)
    names = {ds.get_data_name(i) for i in range(len(ds))}
    assert len(ds) == 3
    assert not any('evt001' in n for n in names)     # the zeroed event dropped
    # the scan is cached (same object on a second call → shared across splits)
    assert read_deposit_counts(step) is read_deposit_counts(step)

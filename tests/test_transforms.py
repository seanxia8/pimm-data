"""Tests for Compose + TRANSFORMS registry + full pipeline.

Ensures raw callables and registry dicts both work in Compose, that
build_dataset resolves by type string, and that transforms + collate
produce the expected batched output keys.
"""

import numpy as np
import pytest
import torch

from pimm_data import (
    Compose, TRANSFORMS, DATASETS, build_dataset,
    JAXTPCDataset, collate_fn,
)


def test_compose_raw_callable():
    c = Compose([lambda d: {**d, 'a': 1}, lambda d: {**d, 'b': 2}])
    out = c({})
    assert out == {'a': 1, 'b': 2}


def test_compose_registry_dict():
    # ToTensor converts numpy arrays in data to tensors
    c = Compose([dict(type='ToTensor')])
    d = {'coord': np.zeros((3, 3), dtype=np.float32)}
    out = c(d)
    assert torch.is_tensor(out['coord'])


def test_compose_mixed():
    marker = {'seen': False}
    def mark(d):
        marker['seen'] = True
        return d
    c = Compose([mark, dict(type='ToTensor')])
    d = {'coord': np.zeros((1, 3), dtype=np.float32)}
    c(d)
    assert marker['seen']


def test_compose_rejects_non_callable_non_dict():
    with pytest.raises(TypeError):
        Compose([42])


def test_transforms_registered_count():
    # 37 from transform.py + PDGToSemantic/RemapSegment/ApplyToStream/
    # AggregateSensorHits from detector_transforms.py (= 41 after the
    # boundary-refactor drops; SSL transforms move to pimm in a later PR).
    assert len(TRANSFORMS) >= 39
    # A few anchor cases
    for name in ('ToTensor', 'GridSample', 'Collect', 'NormalizeCoord',
                 'RandomRotate', 'PDGToSemantic', 'ApplyToStream',
                 'RemapSegment'):
        assert TRANSFORMS.get(name) is not None, f"{name} missing"


def test_build_dataset_resolves_by_type(jaxtpc_data_root):
    cfg = dict(type='JAXTPCDataset', data_root=jaxtpc_data_root,
               split='', dataset_name='sim', modalities=('edep',),
               max_len=2)
    ds = build_dataset(cfg)
    assert isinstance(ds, JAXTPCDataset)
    assert len(ds) == 2


def test_end_to_end_transform_collate(jaxtpc_data_root):
    """Pipeline: ApplyToStream scopes per-cloud transforms, Collect flattens."""
    transform = [
        dict(type='ApplyToStream', stream='edep', transforms=[
            dict(type='NormalizeCoord', center=[0, 0, 0], scale=4000.0),
            dict(type='GridSample', grid_size=0.001, hash_type='fnv',
                 mode='train', return_grid_coord=True),
        ]),
        dict(type='ToTensor'),
        dict(type='Collect', stream='edep',
             keys=('coord', 'grid_coord', 'segment'),
             feat_keys=('coord', 'energy')),
    ]
    ds = JAXTPCDataset(data_root=jaxtpc_data_root, split='',
                       dataset_name='sim',
                       modalities=('edep', 'labl'), label_key='pdg',
                       min_deposits=50, max_len=4, transform=transform)
    batch = collate_fn([ds[0], ds[1]])
    assert batch['coord'].shape[1] == 3
    assert 'offset' in batch
    assert len(batch['offset']) == 2
    assert 'feat' in batch
    assert 'segment' in batch


def test_dataset_getitem_uses_transforms(jaxtpc_data_root):
    """Single-sample path: transforms applied on ds[idx].

    ToTensor recurses into nested sub-dicts, so seg['coord'] becomes a
    tensor in place.
    """
    transform = [dict(type='ToTensor')]
    ds = JAXTPCDataset(data_root=jaxtpc_data_root, split='',
                       dataset_name='sim', modalities=('edep',),
                       max_len=1, transform=transform)
    s = ds[0]
    assert torch.is_tensor(s['edep']['coord'])

"""Pixel-readout tests for JAXTPC.

Exercises:
- readout autodetection from /config.readout_type
- sensor/inst coord dim (3 for pixel) and column names (py/pz/time)
- plane labels are 'volume_{v}_Pixel'
- volume= filter uses Pixel plane
- labl decoration works unchanged for pixel (volume_{v}_Pixel parses OK)
- readers tolerate a missing readout_type attr by inspecting datasets
"""

import os
import h5py
import numpy as np
import pytest

from pimm_data import JAXTPCDataset
from pimm_data.readers.jaxtpc_sensor import JAXTPCSensorReader
from pimm_data.readers.jaxtpc_hits import JAXTPCHitsReader


def test_sensor_reader_detects_pixel(jaxtpc_pixel_data_root):
    r = JAXTPCSensorReader(
        data_root=os.path.join(jaxtpc_pixel_data_root, 'sensor'),
        split='', dataset_name='sim')
    assert r.readout_type == 'pixel'


def test_hits_reader_detects_pixel(jaxtpc_pixel_data_root):
    r = JAXTPCHitsReader(
        data_root=os.path.join(jaxtpc_pixel_data_root, 'hits'),
        split='', dataset_name='sim')
    assert r.readout_type == 'pixel'


def test_pixel_sensor_coord_is_3d(jaxtpc_pixel_data_root):
    ds = JAXTPCDataset(
        data_root=jaxtpc_pixel_data_root, split='',
        modalities=('sensor',))
    s = ds[0]
    assert s['sensor']['readout_type'] == 'pixel'
    assert s['sensor']['coord'].shape[1] == 3
    for plane in s['sensor']['planes']:
        assert plane.endswith('_Pixel')
        cols = s['sensor']['raw'][plane]
        assert set(cols.keys()) == {'py', 'pz', 'time', 'value'}


def test_pixel_hits_coord_is_3d_with_group_id(jaxtpc_pixel_data_root):
    ds = JAXTPCDataset(
        data_root=jaxtpc_pixel_data_root, split='',
        modalities=('hits',))
    s = ds[0]
    assert s['hits']['readout_type'] == 'pixel'
    assert s['hits']['coord'].shape[1] == 3
    for plane in s['hits']['planes']:
        assert plane.endswith('_Pixel')
        cols = s['hits']['raw'][plane]
        assert set(cols.keys()) == {'py', 'pz', 'time', 'charge', 'group_id'}
    # instance column = group_id, per-entry; must match concatenated raw.
    expected = np.concatenate(
        [s['hits']['raw'][p]['group_id'] for p in s['hits']['planes']])
    np.testing.assert_array_equal(s['hits']['instance'], expected.astype(np.int32))


def test_pixel_volume_filter_uses_pixel_plane(jaxtpc_pixel_data_root):
    ds = JAXTPCDataset(
        data_root=jaxtpc_pixel_data_root, split='',
        modalities=('edep', 'sensor', 'hits'), volume=0)
    s = ds[0]
    assert s['sensor']['planes'] == ['volume_0_Pixel']
    assert s['hits']['planes'] == ['volume_0_Pixel']
    # edep is volume-filtered at the reader level too.
    assert set(int(x) for x in s['edep']['volume_id'].ravel()) == {0}


def test_pixel_labl_decoration(jaxtpc_pixel_data_root):
    ds = JAXTPCDataset(
        data_root=jaxtpc_pixel_data_root, split='',
        modalities=('edep', 'hits', 'labl'), label_key='pdg')
    s = ds[0]
    # edep gains instance/segment via labl join
    assert 'instance' in s['edep']
    assert 'segment' in s['edep']
    # hits gains segment via group_to_track → track_pdg chain
    assert 'segment' in s['hits']
    assert s['hits']['segment'].shape == (s['hits']['coord'].shape[0],)
    # not all -1 (synth fixture guarantees matched FKs)
    assert (s['hits']['segment'] != -1).any()
    assert (s['edep']['segment'] != -1).any()


def test_pixel_sensor_values_decode_to_float32(jaxtpc_pixel_data_root):
    """Sensor values must always decode to finite float32 regardless of
    whether the underlying storage was uint16 (digitized) or float32."""
    ds = JAXTPCDataset(
        data_root=jaxtpc_pixel_data_root, split='',
        modalities=('sensor',))
    s = ds[0]
    vals = s['sensor']['energy']
    assert vals.dtype == np.float32
    assert np.isfinite(vals).all()


def test_readers_fall_back_without_readout_type_attr(tmp_path):
    """Readers should still detect pixel when the /config attr is absent,
    by inspecting plane datasets (delta_py)."""
    from pimm_data.testing import make_jaxtpc_sample

    root = str(tmp_path / 'jaxtpc_pixel_noattr')
    make_jaxtpc_sample(root, readout_type='pixel', n_events=1)

    # Strip the readout_type attr from sensor and hits config.
    for mod in ('sensor', 'hits'):
        path = os.path.join(root, mod, f'sim_{mod}_0000.h5')
        with h5py.File(path, 'r+') as f:
            if 'readout_type' in f['config'].attrs:
                del f['config'].attrs['readout_type']

    sr = JAXTPCSensorReader(
        data_root=os.path.join(root, 'sensor'),
        split='', dataset_name='sim')
    ir = JAXTPCHitsReader(
        data_root=os.path.join(root, 'hits'),
        split='', dataset_name='sim')
    assert sr.readout_type == 'pixel'
    assert ir.readout_type == 'pixel'

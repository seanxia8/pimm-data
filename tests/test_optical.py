"""Optical (PMT light) dataset — per-chunk waveforms + the packed second
row-space (REDESIGN §3), for both goop schemas (label_K and east/west). Runs
against synthetic fixtures (or OPTICAL_DATA_ROOT / OPTICAL_EASTWEST_DATA_ROOT).
"""
import numpy as np
import torch

from pimm_data import OpticalDataset, collate_fn
from pimm_data.readers.optical_sensor import (OpticalSensorReader,
                                              OpticalEastWestReader)
from pimm_data.batch_ops import split_event
from pimm_data import _roles


def _collect_cfg():
    return [dict(type='Collect', parts={
        'sensor': dict(
            keys=('pmt_id', 't0_ns', 'length', 'pe', 'instance', 'adc'),
            feat_keys=('pe',),
            offset_keys_dict=dict(offset='pmt_id', wave_offset='adc'),
        )})]


def _ds(root, **kw):
    return OpticalDataset(data_root=root, split='', modalities=('sensor',),
                          max_len=4, transform=_collect_cfg(), **kw)


# --- reader -----------------------------------------------------------------

def test_reader_chunks_are_consistent(optical_data_root):
    r = OpticalSensorReader(data_root=f'{optical_data_root}/sensor',
                            dataset_name='optical')
    assert len(r) > 0 and r.group_kind == 'interaction'
    raw = r.read_event(0)
    K = raw['pmt_id'].shape[0]
    for k in ('t0_ns', 'length', 'pe', 'instance'):
        assert raw[k].shape == (K,)
    assert int(raw['length'].sum()) == raw['adc'].shape[0]   # lossless pack
    assert raw['pmt_id'].max() < r.n_channels
    assert abs(float(raw['adc'].mean())) < 50.0              # pedestal subtracted


def test_dataset_emits_clean_columns_no_coord(optical_data_root):
    sub = _ds(optical_data_root, dataset_name='optical').get_data(0)['sensor']
    K = sub['pmt_id'].shape[0]
    assert 'coord' not in sub                                # dropped (synthetic)
    assert sub['pe'].shape == (K, 1) and sub['instance'].shape == (K,)
    assert sub['adc'].shape[0] == int(sub['length'].sum())
    assert sub['_roles'] == {'adc': ('instance', 'sensor_wave_offset')}


def test_dataset_registered():
    from pimm_data import DATASETS
    assert 'OpticalDataset' in DATASETS


def test_bad_schema_rejected(optical_data_root):
    import pytest
    with pytest.raises(ValueError, match='schema'):
        OpticalDataset(data_root=optical_data_root, schema='nope')


# --- collate: two row-spaces (chunks + packed samples) ----------------------

def test_collate_two_row_spaces(optical_data_root):
    ds = _ds(optical_data_root, dataset_name='optical')
    s0, s1 = ds[0], ds[1]
    batch = collate_fn([s0, s1])

    assert {'sensor_pmt_id', 'sensor_pe', 'sensor_instance', 'sensor_adc',
            'sensor_feat', 'sensor_offset', 'sensor_wave_offset',
            'name', 'split', '_roles'} <= set(batch)
    assert batch['_roles']['sensor_adc'] == ('instance', 'sensor_wave_offset')

    K = s0['sensor_pmt_id'].shape[0] + s1['sensor_pmt_id'].shape[0]
    L = s0['sensor_adc'].shape[0] + s1['sensor_adc'].shape[0]
    assert batch['sensor_pmt_id'].shape[0] == K
    assert batch['sensor_adc'].shape[0] == L
    assert int(batch['sensor_offset'][-1]) == K              # chunks
    assert int(batch['sensor_wave_offset'][-1]) == L         # samples
    assert int(batch['sensor_offset'][-1]) != int(batch['sensor_wave_offset'][-1])


def test_wave_offset_is_not_a_phantom_part(optical_data_root):
    batch = collate_fn([_ds(optical_data_root, dataset_name='optical')[0],
                        _ds(optical_data_root, dataset_name='optical')[1]])
    keys = [k for k in batch if k != '_roles']
    parts = _roles.parts_from_keys(keys, batch['_roles'])
    assert parts == {'sensor'}                               # not {'sensor','sensor_wave'}


def test_split_event_slices_adc_by_wave_offset(optical_data_root):
    ds = _ds(optical_data_root, dataset_name='optical')
    s1 = ds[1]
    batch = collate_fn([ds[0], s1])
    ev1 = split_event(batch, 1)
    assert ev1['sensor_pmt_id'].shape[0] == s1['sensor_pmt_id'].shape[0]
    assert ev1['sensor_adc'].shape[0] == s1['sensor_adc'].shape[0]
    assert int(ev1['sensor_wave_offset'][0]) == s1['sensor_adc'].shape[0]
    assert int(ev1['sensor_offset'][0]) == s1['sensor_pmt_id'].shape[0]
    assert torch.equal(ev1['sensor_adc'], s1['sensor_adc'])
    assert int(ev1['sensor_length'].sum()) == ev1['sensor_adc'].shape[0]


# --- east/west schema -------------------------------------------------------

def test_eastwest_reader_groups_by_side(optical_eastwest_data_root):
    r = OpticalEastWestReader(data_root=f'{optical_eastwest_data_root}/sensor',
                              dataset_name='light')
    assert len(r) > 0 and r.group_kind == 'side'
    raw = r.read_event(0)
    # instance = side index, both sides present -> {0, 1}; pmt_id is per-side
    assert set(np.unique(raw['instance']).tolist()) <= {0, 1}
    assert raw['pmt_id'].max() < r.n_pmts_per_side
    assert int(raw['length'].sum()) == raw['adc'].shape[0]


def test_eastwest_end_to_end(optical_eastwest_data_root):
    ds = _ds(optical_eastwest_data_root, dataset_name='light', schema='east_west')
    batch = collate_fn([ds[0], ds[1]])
    assert batch['_roles']['sensor_adc'] == ('instance', 'sensor_wave_offset')
    assert int(batch['sensor_wave_offset'][-1]) == batch['sensor_adc'].shape[0]
    assert set(np.unique(batch['sensor_instance'].numpy()).tolist()) <= {0, 1}

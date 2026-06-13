"""Optical (PMT light) dataset — per-interaction waveform chunks + the packed
second row-space (REDESIGN §3). Runs against a synthetic ``label_N`` fixture
(or OPTICAL_DATA_ROOT)."""
import numpy as np
import torch

from pimm_data import OpticalDataset, collate_fn
from pimm_data.readers.optical_sensor import OpticalSensorReader
from pimm_data.batch_ops import split_event
from pimm_data import _roles


def _collect_cfg():
    return [dict(type='Collect', modalities={
        'sensor': dict(
            keys=('coord', 'length', 'pe', 'instance', 'adc'),
            feat_keys=('pe',),
            offset_keys_dict=dict(offset='coord', wave_offset='adc'),
        )})]


def _ds(root, **kw):
    return OpticalDataset(data_root=root, split='', dataset_name='optical',
                          modalities=('sensor',), max_len=4, **kw)


# --- reader -----------------------------------------------------------------

def test_reader_chunks_are_consistent(optical_data_root):
    r = OpticalSensorReader(data_root=f'{optical_data_root}/sensor',
                            dataset_name='optical')
    assert len(r) > 0
    raw = r.read_event(0)
    K = raw['pmt_id'].shape[0]
    assert raw['t0_ns'].shape == (K,) and raw['length'].shape == (K,)
    assert raw['pe'].shape == (K,) and raw['interaction'].shape == (K,)
    # packed adc length == sum of per-chunk lengths (lossless pack)
    assert int(raw['length'].sum()) == raw['adc'].shape[0]
    # channels in range; pedestal subtracted (config pedestal=100) -> centered ~0
    assert raw['pmt_id'].max() < r.n_channels
    assert abs(float(raw['adc'].mean())) < 50.0


# --- dataset nested output --------------------------------------------------

def test_dataset_emits_nested_sensor(optical_data_root):
    sub = _ds(optical_data_root).get_data(0)['sensor']
    K = sub['coord'].shape[0]
    assert sub['coord'].shape == (K, 2)                 # [pmt_id, t0_tick]
    assert sub['pe'].shape == (K, 1) and sub['instance'].shape == (K,)
    assert sub['adc'].shape[0] == int(sub['length'].sum())
    assert sub['_roles'] == {'adc': ('instance', 'sensor_wave_offset')}


def test_dataset_registered():
    from pimm_data import DATASETS
    assert 'OpticalDataset' in DATASETS


# --- collate: two row-spaces (chunks + packed samples) ----------------------

def test_collate_two_row_spaces(optical_data_root):
    ds = _ds(optical_data_root, transform=_collect_cfg())
    s0, s1 = ds[0], ds[1]
    batch = collate_fn([s0, s1])

    assert {'sensor_coord', 'sensor_pe', 'sensor_instance', 'sensor_adc',
            'sensor_feat', 'sensor_offset', 'sensor_wave_offset',
            'name', 'split', '_roles'} <= set(batch)
    assert batch['_roles']['sensor_adc'] == ('instance', 'sensor_wave_offset')

    K = s0['sensor_coord'].shape[0] + s1['sensor_coord'].shape[0]
    L = s0['sensor_adc'].shape[0] + s1['sensor_adc'].shape[0]
    # chunk row-space vs sample row-space — distinct cumulative offsets (B,)
    assert batch['sensor_coord'].shape[0] == K
    assert batch['sensor_adc'].shape[0] == L
    assert batch['sensor_offset'].tolist()[-1] == K
    assert batch['sensor_wave_offset'].tolist()[-1] == L
    # adc is NOT in the chunk (point) row-space
    assert batch['sensor_offset'].tolist()[-1] != batch['sensor_wave_offset'].tolist()[-1]


def test_wave_offset_is_not_a_phantom_part(optical_data_root):
    ds = _ds(optical_data_root, transform=_collect_cfg())
    batch = collate_fn([ds[0], ds[1]])
    keys = [k for k in batch if k != '_roles']
    parts = _roles.parts_from_keys(keys, batch['_roles'])
    assert parts == {'sensor'}                          # NOT {'sensor','sensor_wave'}


# --- split_event slices the packed waveform by the sample offset ------------

def test_split_event_slices_adc_by_wave_offset(optical_data_root):
    ds = _ds(optical_data_root, transform=_collect_cfg())
    s0, s1 = ds[0], ds[1]
    batch = collate_fn([s0, s1])

    ev1 = split_event(batch, 1)
    # chunks of event 1 come back, packed adc sliced by the SAMPLE span (not chunks)
    assert ev1['sensor_coord'].shape[0] == s1['sensor_coord'].shape[0]
    assert ev1['sensor_adc'].shape[0] == s1['sensor_adc'].shape[0]
    assert int(ev1['sensor_wave_offset'][0]) == s1['sensor_adc'].shape[0]
    assert int(ev1['sensor_offset'][0]) == s1['sensor_coord'].shape[0]
    # byte-identical to the pre-collate single event
    assert torch.equal(ev1['sensor_adc'], s1['sensor_adc'])
    # per-chunk waveform reconstruction still works: Σ(length) spans adc
    assert int(ev1['sensor_length'].sum()) == ev1['sensor_adc'].shape[0]

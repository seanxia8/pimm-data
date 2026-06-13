"""Campaign data-loader recipes load and yield the expected challenge batch.

Each recipe in ``configs/<dataset>/<challenge>.py`` is the DATA-LOADING half of a
campaign challenge (CAMPAIGN.md). This gate execs the recipe, points its train
dataset at the synthetic fixture, runs collate, and asserts the flat-prefixed
keys the challenge needs. It keeps the recipes honest as the campaign fans out.
"""
import copy
import os
import runpy

from pimm_data import build_dataset, collate_fn

_CONFIGS = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs')


def _recipe(rel):
    return runpy.run_path(os.path.join(_CONFIGS, rel))['data']


def _build(spec, **override):
    spec = dict(copy.deepcopy(spec))
    spec.update(override)
    return build_dataset(spec)


def _collate(ds):
    b = collate_fn([ds[0], ds[1]])
    return {k for k in b if k != '_roles'}, b


def test_jaxtpc_semseg_recipe_loads(jaxtpc_data_root):
    d = _recipe('jaxtpc/semseg_5cls.py')
    ds = _build(d['train'], data_root=jaxtpc_data_root, split='', min_deposits=0)
    keys, b = _collate(ds)
    assert {'step_coord', 'step_grid_coord', 'step_segment', 'step_feat',
            'step_offset'} <= keys
    assert b['step_feat'].shape[1] == 4               # [coord|energy]


def test_lucid_perpmt_seg_recipe_loads(lucid_data_root):
    d = _recipe('lucid/perpmt_seg_hits.py')
    ds = _build(d['train'], data_root=lucid_data_root, split='')
    keys, b = _collate(ds)
    assert {'hits_coord', 'hits_segment', 'hits_instance', 'hits_feat',
            'hits_offset'} <= keys
    assert b['hits_feat'].shape[1] == 5               # [coord|energy|time]


def test_optical_interaction_recipe_loads(optical_data_root):
    d = _recipe('optical/interaction_discrimination.py')
    ds = _build(d['train'], data_root=optical_data_root,
                dataset_name='optical', split='')
    keys, b = _collate(ds)
    assert {'sensor_pmt_id', 'sensor_adc', 'sensor_instance', 'sensor_offset',
            'sensor_wave_offset'} <= keys
    assert b['_roles']['sensor_adc'] == ('instance', 'sensor_wave_offset')
    # two row-spaces: chunks vs packed samples
    assert int(b['sensor_offset'][-1]) != int(b['sensor_wave_offset'][-1])

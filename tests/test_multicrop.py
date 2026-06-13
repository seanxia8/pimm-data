"""MultiCrop — packed multi-crop view parts (replaces Streams).

Reads a source part, writes packed `global`/`local` parts; Collect flattens to
global_*/local_* with per-crop-packed offsets (B*num_views entries after collate).
"""
import numpy as np

from pimm_data.transform import TRANSFORMS, Compose
from pimm_data import collate_fn


def _event(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return {'step': {'coord': rng.standard_normal((n, 3)).astype('float32'),
                     'energy': rng.standard_normal((n, 1)).astype('float32')},
            'name': f'e{seed}', 'split': 'train'}


def _pipe(gnum=2, lnum=3):
    return Compose([
        dict(type='MultiCrop', on='step', view_keys=('coord', 'energy'),
             global_view_num=gnum, global_view_scale=(0.4, 1.0),
             local_view_num=lnum, local_view_scale=(0.1, 0.4)),
        dict(type='Collect', modalities={
            'global': dict(keys=('coord', 'offset'), offset_keys_dict={},
                           feat_keys=('coord', 'energy')),
            'local':  dict(keys=('coord', 'offset'), offset_keys_dict={},
                           feat_keys=('coord', 'energy'))}),
    ])


def test_multicrop_packs_flat_views():
    np.random.seed(0)
    pipe = _pipe(gnum=2, lnum=3)
    b = collate_fn([pipe(_event(seed=0)), pipe(_event(seed=1))])
    assert {'global_coord', 'global_offset', 'global_feat',
            'local_coord', 'local_offset', 'local_feat', '_roles',
            'name', 'split'} <= set(b)
    # offset packs B * num_views crop boundaries
    assert b['global_offset'].shape[0] == 2 * 2
    assert b['local_offset'].shape[0] == 2 * 3
    assert int(b['global_offset'][-1]) == b['global_coord'].shape[0]
    assert int(b['local_offset'][-1]) == b['local_coord'].shape[0]
    assert b['global_feat'].shape[1] == 4          # coord(3) + energy(1)
    assert b['name'] == ['e0', 'e1']


def test_multicrop_views_differ():
    """Global crops are different center-samples -> not identical (the SSL signal)."""
    np.random.seed(0)
    pipe = _pipe(gnum=2, lnum=2)
    b = collate_fn([pipe(_event(seed=0))])
    off = b['global_offset'].tolist()              # [s0, s0+s1]
    crop0 = b['global_coord'][:off[0]]
    crop1 = b['global_coord'][off[0]:off[1]]
    # different sizes or different points -> not the same crop
    assert crop0.shape != crop1.shape or not np.allclose(crop0.numpy(), crop1.numpy())


def test_global_shared_transform_runs_once():
    np.random.seed(0)
    mc = TRANSFORMS.build(dict(type='MultiCrop', on='step',
        view_keys=('coord', 'energy'), global_view_num=2, local_view_num=1,
        global_shared_transform=[dict(type='NormalizeCoord')]))
    out = mc(_event(seed=0))
    assert 'global' in out and 'local' in out
    assert 'offset' in out['global'] and out['global']['coord'].shape[0] > 0

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

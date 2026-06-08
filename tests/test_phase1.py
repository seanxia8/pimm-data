"""Phase 1 — labels (labl -> labels=), G1 seeding, G2 name pass-through."""
import numpy as np

from pimm_data.transform import Collect


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

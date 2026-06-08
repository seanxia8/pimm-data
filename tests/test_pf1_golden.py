"""PF1 golden snapshots — lock single-modality byte-identity + label decoration.

These fingerprints are captured from the CURRENT code on the synthetic (seed=0)
fixture. They make the RESTRUCTURE plan's two breaking phases verifiable:

* Phase 0 (``stream``→``modality`` rename): pure renaming — the produced tensors
  must be **byte-identical**.
* Phase 1 (``labl`` out of ``modalities=`` → ``labels=``, direct FK joins): the API
  changes but the decorated ``segment``/``instance`` values must be **unchanged**.

After those phases, migrate the dataset-build calls below to the new API
(``modalities=``/``labels=``, ``Collect(modality=)``) — the frozen fingerprints
MUST still match. A diff here means the refactor changed data, not just names.

The hashes are specific to the synthetic seed=0 fixture, so the full comparison
is gated on ``jaxtpc_is_synthetic``; on a real-data override only the structural
shape/dtype is checked.
"""

import hashlib

import numpy as np
import pytest
import torch

from pimm_data import JAXTPCDataset, collate_fn

# --- frozen fingerprints (captured from current behavior, synthetic seed=0) ---
# value = (shape, dtype, sha1(bytes)[:16]) for arrays; repr(...) for metadata.
C1_BATCH = {
    'coord': ((240, 3), 'float32', 'ec942392e142f1f7'),
    'feat':  ((240, 4), 'float32', '8194f574fbe936da'),
    'name':  "['sim_step_0000.h5_evt000', 'sim_step_0000.h5_evt001']",
    'offset': ((2,), 'int64', '2bf2fc17c20c1d7d'),
    'split': "['', '']",
}
C2_BATCH = {
    'coord': ((240, 3), 'float32', 'ec942392e142f1f7'),
    'feat':  ((240, 4), 'float32', '8194f574fbe936da'),
    'name':  "['sim_step_0000.h5_evt000', 'sim_step_0000.h5_evt001']",
    'offset': ((2,), 'int64', '2bf2fc17c20c1d7d'),
    'segment': ((240,), 'int32', 'd7b8e4c2fcd7506c'),
    'split': "['', '']",
}
STEP_SEG = ((120,), 'int32', '13e0b23826ddf03e')
STEP_INST = ((120,), 'int32', 'bbc81806166df68d')
HITS_SEG = ((240,), 'int32', 'd359c48d76290f76')
HITS_INST = ((240,), 'int32', '1e38e79c346f5729')


def _fp(x):
    if isinstance(x, torch.Tensor):
        a = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        a = x
    else:
        return repr(x)
    a = np.ascontiguousarray(a)
    return (tuple(a.shape), str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest()[:16])


def _fps(d):
    return {k: _fp(v) for k, v in sorted(d.items()) if not isinstance(v, dict)}


def _ds(root, modalities, transform=None, **kw):
    return JAXTPCDataset(data_root=root, split='', dataset_name='sim',
                         modalities=modalities, label_key='pdg', min_deposits=0,
                         max_len=2, transform=transform, **kw)


def _structural(fp_tuple):
    """(shape, dtype) only — for the real-data path where hashes won't match."""
    return fp_tuple[:2] if isinstance(fp_tuple, tuple) else fp_tuple


def _assert(actual, expected, synthetic):
    if synthetic:
        assert actual == expected
    else:
        # real-data override: hashes/shapes differ, only lock dtype layout
        a = {k: (v[1] if isinstance(v, tuple) else v) for k, v in actual.items()}
        e = {k: (v[1] if isinstance(v, tuple) else v) for k, v in expected.items()}
        assert a == e


def test_pf1_c1_step_ssl_batch(jaxtpc_data_root, jaxtpc_is_synthetic):
    ds = _ds(jaxtpc_data_root, ('step',),
             transform=[dict(type='Collect', modality='step',
                             keys=('coord',), feat_keys=('coord', 'energy'))])
    batch = collate_fn([ds[0], ds[1]])
    _assert(_fps(batch), C1_BATCH, jaxtpc_is_synthetic)


def test_pf1_c2_step_seg_batch(jaxtpc_data_root, jaxtpc_is_synthetic):
    ds = _ds(jaxtpc_data_root, ('step', 'labl'),
             transform=[dict(type='Collect', modality='step',
                             keys=('coord', 'segment'),
                             feat_keys=('coord', 'energy'))])
    batch = collate_fn([ds[0], ds[1]])
    _assert(_fps(batch), C2_BATCH, jaxtpc_is_synthetic)


def test_pf1_label_decoration_parity(jaxtpc_data_root, jaxtpc_is_synthetic):
    step = _ds(jaxtpc_data_root, ('step', 'labl')).get_data(0)['step']
    hits = _ds(jaxtpc_data_root, ('hits', 'labl')).get_data(0)['hits']
    if jaxtpc_is_synthetic:
        assert _fp(step['segment']) == STEP_SEG
        assert _fp(step['instance']) == STEP_INST
        assert _fp(hits['segment']) == HITS_SEG
        assert _fp(hits['instance']) == HITS_INST
    else:
        assert _structural(_fp(step['segment'])) == _structural(STEP_SEG)
        assert _structural(_fp(hits['instance'])) == _structural(HITS_INST)

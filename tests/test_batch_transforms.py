"""Tests for the post-collate dense GPU path (dense_ops + batch_transforms).

Covers: densify torch↔numpy bit-exact (CPU + CUDA), offset2batch edges, coherent
bit-exact vs the numpy oracle / JAXTPC, incoherent statistical RMS, digitize
bit-exact, the end-to-end runner (Densify→AddNoise→Digitize) on a collated batch,
empty-stages no-op, and the no-`torch.cuda`-import gate.
"""

import os
from copy import deepcopy

import numpy as np
import pytest
import torch

from pimm_data import (JAXTPCDataset, collate_fn,
                       build_sensor_gpu_stages, move_to_device)
from pimm_data.transform import Compose
from pimm_data import dense_ops
from pimm_data.jaxtpc import canonical_plane_id
from pimm_data.detector_transforms import Densify
from pimm_data.noise import coherent_noise as coherent_noise_np, digitize as digitize_np

_WL = {'U': (0.42, 4.63), 'V': (0.42, 4.63), 'Y': (2.33, 2.33)}


def _ds(root, B=2):
    return JAXTPCDataset(
        data_root=root, split='', dataset_name='sim', modalities=('sensor',),
        max_len=B, wire_lengths_per_plane=_WL,
        transform=[dict(type='Collect', part='sensor',
                        keys=('wire', 'time', 'value', 'plane_gid'))])


def _sensor_batch(root, B=2):
    ds = _ds(root, B)
    batch = collate_fn([ds[i] for i in range(B)])
    ds.get_data(0)  # ensure reader geometry is populated for plane_geometry()
    return ds, batch


# --------------------------------------------------------------------------
# densify
# --------------------------------------------------------------------------

def test_offset2batch_basic_and_empty():
    # cumulative offsets: sample0=3 hits, sample1=0 (empty), sample2=2
    assert dense_ops.offset2batch(torch.tensor([3, 3, 5])).tolist() == [0, 0, 0, 2, 2]
    assert dense_ops.offset2batch(torch.tensor([3])).tolist() == [0, 0, 0]   # B=1
    assert dense_ops.offset2batch(torch.tensor([0, 0, 0])).tolist() == []     # all-empty


def test_densify_torch_matches_numpy_cpu(jaxtpc_data_root):
    ds, batch = _sensor_batch(jaxtpc_data_root, B=2)
    geom = ds.plane_geometry()
    grids = dense_ops.densify(batch['wire'], batch['time'], batch['value'],
                              batch['plane_gid'], batch['offset'], geom)
    # numpy oracle: run the per-sample Densify on each event, compare per plane
    for i in range(2):
        sub = ds.get_data(i)['sensor']
        Densify()(sub)
        for label, img in sub['dense'].items():
            gid = canonical_plane_id(label)
            assert np.array_equal(grids[gid][i].cpu().numpy(), img), \
                f"event {i} plane {label}: torch densify != numpy Densify"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_densify_cpu_matches_cuda(jaxtpc_data_root):
    ds, batch = _sensor_batch(jaxtpc_data_root, B=2)
    geom = ds.plane_geometry()
    cpu = dense_ops.densify(batch['wire'], batch['time'], batch['value'],
                            batch['plane_gid'], batch['offset'], geom)
    cu = move_to_device(batch, 'cuda')
    gpu = dense_ops.densify(cu['wire'], cu['time'], cu['value'],
                            cu['plane_gid'], cu['offset'], geom)
    for gid in cpu:
        assert torch.equal(cpu[gid], gpu[gid].cpu())


def test_densify_rejects_float_indices():
    geom = {0: {'n_wires': 8, 'n_ticks': 16}}
    with pytest.raises(TypeError):
        dense_ops.densify(torch.zeros(3), torch.zeros(3), torch.ones(3),
                          torch.zeros(3, dtype=torch.long), torch.tensor([3]), geom)


# --------------------------------------------------------------------------
# noise — coherent bit-exact, incoherent statistical
# --------------------------------------------------------------------------

def test_coherent_batched_matches_numpy_oracle():
    # the bit-exact numpy oracle is now opt-in (coherent_numpy=True); the default
    # path is the on-device torch port (statistical parity only — see below).
    W, T, gid, seed = 200, 2048, 0, 12345
    geom = {gid: {'n_wires': W, 'n_ticks': T}}
    grids = {gid: torch.zeros(1, W, T)}
    dense_ops.add_intrinsic_noise(grids, geom, seeds=[seed], coherent=True,
                                  incoherent=False, group_size=64, coh_rms=2.5,
                                  coherent_numpy=True)
    expected = coherent_noise_np(W, T, np.random.default_rng(seed),
                                 group_size=64, rms_adc=2.5)
    assert np.allclose(grids[gid][0].numpy(), expected, atol=1e-5)


def test_coherent_torch_default_statistical():
    """Default (on-device torch) coherent: not bit-exact to numpy, but matches the
    forward model's structure — within-group identical, per-group RMS == coh_rms,
    adjacent-group anti-correlation ~ -2β/(1+2β²)."""
    W, T, gid, gs, rms, beta = 256, 4096, 0, 64, 2.5, 0.15
    geom = {gid: {'n_wires': W, 'n_ticks': T}}
    grids = {gid: torch.zeros(1, W, T)}
    dense_ops.add_intrinsic_noise(grids, geom, seeds=[3], coherent=True,
                                  incoherent=False, group_size=gs, coh_rms=rms,
                                  beta=beta)
    x = grids[gid][0].numpy()
    g0 = x[:gs]                                      # within-group identical (common mode)
    assert np.abs(g0 - g0[0]).max() < 1e-5
    reps = x[::gs]                                   # one wire per group
    assert abs(reps.std(axis=1).mean() - rms) < 0.2  # per-group RMS ~ coh_rms
    lag1 = np.corrcoef(reps[:-1].ravel(), reps[1:].ravel())[0, 1]
    assert abs(lag1 - (-2 * beta / (1 + 2 * beta ** 2))) < 0.05


def test_incoherent_rms_matches_enc_statistical():
    W, T, gid = 64, 8192, 0
    L = np.linspace(0.42, 4.63, W).astype(np.float32)
    geom = {gid: {'n_wires': W, 'n_ticks': T, 'wire_lengths': L}}
    grids = {gid: torch.zeros(1, W, T)}
    x, y, z = 0.90, 0.79, 0.22
    dense_ops.add_intrinsic_noise(grids, geom, seeds=[7], coherent=False,
                                  incoherent=True, enc=(x, y, z))
    rms = grids[gid][0].std(dim=1, unbiased=False).numpy()
    expected = np.sqrt(x**2 + (y + z * L) ** 2)
    assert abs((rms / expected).mean() - 1.0) < 0.02       # mean tight
    assert float(np.abs(rms / expected - 1.0).max()) < 0.15  # per-channel band


def test_incoherent_requires_wire_lengths():
    geom = {0: {'n_wires': 8, 'n_ticks': 64}}  # no wire_lengths
    grids = {0: torch.zeros(1, 8, 64)}
    with pytest.raises(ValueError):
        dense_ops.add_intrinsic_noise(grids, geom, seeds=[0], coherent=False,
                                      incoherent=True)


def test_noise_reproducible_by_seed():
    W, T, gid = 32, 1024, 0
    L = np.full(W, 2.33, np.float32)
    geom = {gid: {'n_wires': W, 'n_ticks': T, 'wire_lengths': L}}

    def run(seed):
        g = {gid: torch.zeros(1, W, T)}
        dense_ops.add_intrinsic_noise(g, geom, seeds=[seed], coherent=True,
                                      incoherent=True)
        return g[gid].clone()

    assert torch.equal(run(99), run(99))      # same seed -> identical
    assert not torch.equal(run(1), run(2))    # different seed -> different


# --------------------------------------------------------------------------
# digitize
# --------------------------------------------------------------------------

def test_digitize_matches_numpy():
    g = (torch.randn(2, 16, 32) * 1000.0)
    out = dense_ops.digitize({0: g}, pedestal=410, n_bits=12)
    exp = digitize_np(g.numpy(), 410, n_bits=12)
    assert np.array_equal(out[0].numpy(), exp)


# --------------------------------------------------------------------------
# dense chain — end to end (Compose; no runner)
# --------------------------------------------------------------------------

def test_dense_chain_end_to_end(jaxtpc_data_root):
    ds, batch = _sensor_batch(jaxtpc_data_root, B=2)
    geom = ds.plane_geometry()
    stages = build_sensor_gpu_stages(geom, device='cpu', coherent=True,
                                     incoherent=True, n_bits=12)
    out = stages(deepcopy(batch))                       # a runnable Compose
    assert 'dense' in out
    for gid, g in out['dense'].items():
        assert g.shape[0] == 2
        assert g.shape[1:] == (geom[gid]['n_wires'], geom[gid]['n_ticks'])
        ped = geom[gid].get('pedestal', 0)
        assert torch.allclose(g, torch.round(g))        # digitized
        assert g.min() >= -ped - 1e-3 and g.max() <= 4095 - ped + 1e-3


def test_dense_per_event_reproducible(jaxtpc_data_root):
    ds, batch = _sensor_batch(jaxtpc_data_root, B=2)
    geom = ds.plane_geometry()
    mk = lambda seed: build_sensor_gpu_stages(geom, device='cpu', base_seed=seed,
                                              coherent=True, incoherent=False)
    a = mk(5)(deepcopy(batch))                          # noise self-seeds from name
    b = mk(5)(deepcopy(batch))
    c = mk(6)(deepcopy(batch))
    g0 = sorted(a['dense'])[0]
    assert torch.equal(a['dense'][g0], b['dense'][g0])
    assert not torch.equal(a['dense'][g0], c['dense'][g0])


def test_todevice_only_is_move_no_dense(jaxtpc_data_root):
    _, batch = _sensor_batch(jaxtpc_data_root, B=2)
    out = Compose([dict(type='ToDevice', device='cpu')])(deepcopy(batch))
    assert 'dense' not in out
    assert torch.equal(out['wire'], batch['wire'])


def test_move_to_device_idempotent():
    b = {'x': torch.zeros(3), 'name': ['a', 'b'], 'meta': {'y': torch.ones(2)}}
    out = move_to_device(b, 'cpu')
    assert out['x'].device.type == 'cpu' and out['meta']['y'].device.type == 'cpu'
    assert out['name'] == ['a', 'b']


# --------------------------------------------------------------------------
# invariant: pimm-data never imports torch.cuda
# --------------------------------------------------------------------------

def _import_jaxtpc():
    import sys
    for root in (os.environ.get('JAXTPC_ROOT'), '/sdf/group/neutrino/omara/JAXTPC'):
        if root and os.path.isdir(os.path.join(root, 'tools')):
            if root not in sys.path:
                sys.path.insert(0, root)
            # evict a foreign cached `tools` (namespace collision) so this
            # resolves to JAXTPC's parity oracle regardless of import order.
            for m in [k for k in list(sys.modules)
                      if k == 'tools' or k.startswith('tools.')]:
                f = getattr(sys.modules[m], '__file__', None) or ''
                if not f.startswith(root):
                    del sys.modules[m]
            try:
                import tools.coherent_noise as cn
            except Exception:
                return None
            if not (getattr(cn, '__file__', '') or '').startswith(root):
                return None
            return cn
    return None


# --------------------------------------------------------------------------
# de-tautologized: BATCHED coherent path vs JAXTPC (not the pimm-data oracle)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("T", [2048, 4321])  # even AND odd (production num_time is odd)
def test_coherent_batched_matches_jaxtpc(T):
    cn = _import_jaxtpc()
    if cn is None:
        pytest.skip("JAXTPC not importable")
    W, gid, seed = 200, 0, 777
    geom = {gid: {'n_wires': W, 'n_ticks': T}}
    grids = {gid: torch.zeros(1, W, T)}
    # bit-exactness to JAXTPC is the numpy oracle's contract (coherent_numpy=True);
    # the default torch path is statistical-parity only.
    dense_ops.add_intrinsic_noise(grids, geom, seeds=[seed], coherent=True,
                                  incoherent=False, group_size=64, coh_rms=2.5,
                                  coherent_numpy=True)
    expected = cn.generate_coherent_noise(
        n_wires=W, n_ticks=T, group_size=64, beta=0.15, rms_adc=2.5,
        corner_freq_hz=20000.0, spectral_slope=1.5, sampling_rate_hz=2e6,
        rng=np.random.default_rng(seed))
    assert np.allclose(grids[gid][0].numpy(), expected, atol=1e-5)


@pytest.mark.parametrize("T", [2048, 4321])
def test_incoherent_rms_at_odd_and_even_T(T):
    W, gid = 48, 0
    L = np.linspace(0.42, 4.63, W).astype(np.float32)
    geom = {gid: {'n_wires': W, 'n_ticks': T, 'wire_lengths': L}}
    grids = {gid: torch.zeros(1, W, T)}
    x, y, z = 0.90, 0.79, 0.22
    dense_ops.add_intrinsic_noise(grids, geom, seeds=[3], coherent=False,
                                  incoherent=True, enc=(x, y, z))
    rms = grids[gid][0].std(dim=1, unbiased=False).numpy()
    expected = np.sqrt(x**2 + (y + z * L) ** 2)
    assert abs((rms / expected).mean() - 1.0) < 0.03


# --------------------------------------------------------------------------
# per-event seed -> per-event noise (within one batch)
# --------------------------------------------------------------------------

def test_within_batch_events_differ_by_seed():
    W, T, gid = 32, 512, 0
    geom = {gid: {'n_wires': W, 'n_ticks': T}}
    g = {gid: torch.zeros(2, W, T)}
    dense_ops.add_intrinsic_noise(g, geom, seeds=[11, 22], coherent=True, incoherent=False)
    assert not torch.equal(g[gid][0], g[gid][1])          # distinct seeds -> distinct noise
    g2 = {gid: torch.zeros(2, W, T)}
    dense_ops.add_intrinsic_noise(g2, geom, seeds=[11, 11], coherent=True, incoherent=False)
    assert torch.equal(g2[gid][0], g2[gid][1])            # equal seeds -> equal noise


# --------------------------------------------------------------------------
# canonical plane_id: stable under volume filter / empty plane; densify guards
# --------------------------------------------------------------------------

def test_densify_under_volume_filter(jaxtpc_data_root):
    ds = JAXTPCDataset(data_root=jaxtpc_data_root, split='', dataset_name='sim',
                       modalities=('sensor',), volume=1, max_len=2,
                       wire_lengths_per_plane=_WL,
                       transform=[dict(type='Collect', part='sensor',
                                       keys=('wire', 'time', 'value', 'plane_gid'))])
    batch = collate_fn([ds[0], ds[1]])
    ds.get_data(0)
    geom = ds.plane_geometry()
    assert geom and set(geom) <= {3, 4, 5}, "volume=1 -> canonical gids 3,4,5 only"
    grids = dense_ops.densify(batch['wire'], batch['time'], batch['value'],
                              batch['plane_gid'], batch['offset'], geom)
    assert set(grids) <= {3, 4, 5}


def test_densify_length_mismatch_raises():
    geom = {0: {'n_wires': 8, 'n_ticks': 16}}
    with pytest.raises(ValueError):
        dense_ops.densify(torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long),
                          torch.ones(3), torch.zeros(3, dtype=torch.long),
                          torch.tensor([3]), geom)


def test_densify_missing_plane_in_registry_raises():
    geom = {0: {'n_wires': 8, 'n_ticks': 16}}  # registry lacks plane 5
    with pytest.raises(KeyError):
        dense_ops.densify(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
                          torch.ones(2), torch.tensor([5, 5]), torch.tensor([2]), geom)


def test_densify_out_of_bounds_raises():
    geom = {0: {'n_wires': 4, 'n_ticks': 4}}
    with pytest.raises(ValueError):
        dense_ops.densify(torch.tensor([10]), torch.tensor([0]), torch.ones(1),
                          torch.tensor([0]), torch.tensor([1]), geom)


def test_canonical_plane_id_rejects_malformed():
    from pimm_data.jaxtpc import canonical_plane_id
    for bad in ('east_U', 'volume_0', 'volumeX_0_U', 'volume_0_Q'):
        with pytest.raises(ValueError):
            canonical_plane_id(bad)


# --------------------------------------------------------------------------
# CUDA coverage (skip if no GPU): noise + digitize + end-to-end on device
# --------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_coherent_cpu_matches_cuda():
    # Only the numpy oracle (coherent_numpy=True) is device-independent. The default
    # torch coherent uses torch RNG, whose CPU/CUDA streams differ by design, so it
    # is device-specific (like the incoherent port) — not tested for cross-device equality.
    W, T, gid = 64, 1024, 0
    geom = {gid: {'n_wires': W, 'n_ticks': T}}
    out = {}
    for dev in ('cpu', 'cuda'):
        g = {gid: torch.zeros(1, W, T, device=dev)}
        dense_ops.add_intrinsic_noise(g, geom, seeds=[5], coherent=True,
                                      incoherent=False, coherent_numpy=True)
        out[dev] = g[gid].cpu()
    assert torch.allclose(out['cpu'], out['cuda'], atol=1e-5)  # numpy oracle -> device-independent


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_incoherent_rms_on_cuda():
    W, T, gid = 48, 4096, 0
    L = np.linspace(0.42, 4.63, W).astype(np.float32)
    geom = {gid: {'n_wires': W, 'n_ticks': T, 'wire_lengths': L}}
    g = {gid: torch.zeros(1, W, T, device='cuda')}
    x, y, z = 0.90, 0.79, 0.22
    dense_ops.add_intrinsic_noise(g, geom, seeds=[3], coherent=False, incoherent=True,
                                  enc=(x, y, z))
    rms = g[gid][0].std(dim=1, unbiased=False).cpu().numpy()
    expected = np.sqrt(x**2 + (y + z * L) ** 2)
    assert abs((rms / expected).mean() - 1.0) < 0.05  # statistical (cuFFT/RNG != numpy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_end_to_end_on_cuda(jaxtpc_data_root):
    ds, batch = _sensor_batch(jaxtpc_data_root, B=2)
    geom = ds.plane_geometry()
    stages = build_sensor_gpu_stages(geom, device='cuda', coherent=True,
                                     incoherent=True, n_bits=12)
    out = stages(deepcopy(batch))
    for gid, g in out['dense'].items():
        assert g.device.type == 'cuda'
        assert torch.allclose(g, torch.round(g))  # digitized


def test_no_torch_cuda_import_in_source():
    import pimm_data
    root = os.path.dirname(pimm_data.__file__)
    offenders = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.endswith('.py'):
                with open(os.path.join(dirpath, fn)) as f:
                    src = f.read()
                if 'import torch.cuda' in src or 'from torch.cuda' in src:
                    offenders.append(os.path.relpath(os.path.join(dirpath, fn), root))
    assert not offenders, f"torch.cuda imported in: {offenders}"

"""Guard the dense-op twin collapse: the post-collate torch core
(:mod:`pimm_data.dense_ops`) must reproduce the pre-collate numpy transforms
bit-for-bit on a single event BEFORE the numpy twins are deleted. This witnesses
that the collapse is lossless — densify is one scope='sample' op, placed by the
harness, not two implementations.
"""
import numpy as np
import torch

from pimm_data import dense_ops
from pimm_data.detector_transforms import Densify
from pimm_data.noise import digitize as digitize_np


def _synthetic_plane(W, T, n, seed):
    """n unique (wire,time) cells in a W×T grid with float values."""
    rng = np.random.default_rng(seed)
    flat = rng.choice(W * T, size=n, replace=False)
    wire = (flat // T).astype(np.int64)
    time = (flat % T).astype(np.int64)
    value = rng.standard_normal(n).astype(np.float32)
    return wire, time, value


def test_densify_oracle_equiv_single_event():
    """numpy Densify (per-event, nested-by-label) == dense_ops.densify (flat COO,
    B=1) for the same event, bit-for-bit."""
    W, T, gid = 13, 29, 0
    wire, time, value = _synthetic_plane(W, T, 40, seed=1)

    # pre-collate numpy path (nested-by-label representation)
    sub = {'readout_type': 'wire',
           'raw': {'U': {'wire': wire, 'time': time, 'value': value}},
           'shape': {'U': (W, T)}}
    np_grid = Densify()(sub)['dense']['U']                       # (W, T)

    # post-collate torch path (flat COO, single event => B=1, offset=[N])
    grids = dense_ops.densify(
        torch.from_numpy(wire), torch.from_numpy(time), torch.from_numpy(value),
        torch.full((wire.size,), gid, dtype=torch.long),
        torch.tensor([wire.size]), {gid: {'n_wires': W, 'n_ticks': T}})
    torch_grid = grids[gid][0].numpy()                           # (W, T)

    assert torch_grid.shape == np_grid.shape == (W, T)
    assert np.array_equal(torch_grid, np_grid)


def test_digitize_oracle_equiv():
    """dense_ops.digitize == noise.digitize (the oracle numpy Digitize calls)."""
    W, T, gid, ped = 11, 17, 0, 400.0
    rng = np.random.default_rng(2)
    g = (rng.standard_normal((W, T)) * 50).astype(np.float32)

    np_out = digitize_np(g, ped, n_bits=12, gain=1.0)
    torch_out = dense_ops.digitize(
        {gid: torch.from_numpy(g)}, ped, n_bits=12, gain=1.0)[gid].numpy()

    assert np.array_equal(torch_out, np_out)

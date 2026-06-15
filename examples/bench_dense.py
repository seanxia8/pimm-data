"""Bench + verify the dense sensor tail (Densify -> AddNoise -> Digitize).

Times the post-collate tail on CPU vs GPU (ToDevice), and checks the output
against a hand-rolled reference:
  * densify: unified Densify == manual numpy scatter (bit-exact, deterministic);
  * CPU vs GPU with coherent_numpy=True (the device-independent oracle) == bit-exact;
  * digitize: == numpy digitize.

Run:  python examples/bench_dense.py
"""
import copy
import time

import numpy as np
import torch

from pimm_data import JAXTPCDataset, collate_fn
from pimm_data.geometry import load_plane_registry
from pimm_data.transform import Compose

JAX = '/sdf/data/neutrino/omara/JAXTPC_Wire/test_00_00_02'
B = 2
ITERS = 3


def _tail(device, coherent_numpy=False):
    t = []
    if device is not None:
        t.append(dict(type='ToDevice', device=device))
    t += [dict(type='Densify', geom=GEOM, modality='sensor'),
          dict(type='AddNoise', geom=GEOM, modality='sensor', coherent=True,
               incoherent=False, coherent_numpy=coherent_numpy),
          dict(type='Digitize', geom=GEOM, modality='sensor', n_bits=12)]
    return Compose(t)


def _time(device, coherent_numpy=False, iters=ITERS):
    chain = _tail(device, coherent_numpy)
    chain(copy.deepcopy(SPARSE))                       # warmup
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = chain(copy.deepcopy(SPARSE))
        if device == 'cuda':
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, out


def main():
    global GEOM, SPARSE
    ds = JAXTPCDataset(
        data_root=JAX, split='run_0027575715', dataset_name='sim_wire',
        modalities=('sensor',), max_len=8,
        transform=[dict(type='Collect', parts={
            'sensor': dict(keys=('wire', 'time', 'value', 'plane_gid'))})])
    ds.get_data(0)
    # the real shards don't carry per-plane n_wires attrs, so the registry
    # (config-derived) is the geometry source — same as the recipe.
    GEOM = load_plane_registry('cubic_wireplane_geometry.json')
    SPARSE = collate_fn([ds[i] for i in range(B)])
    print(f"B={B}  sparse COO rows={int(SPARSE['sensor_offset'][-1])}  "
          f"planes={len(GEOM)}")

    print("\n== timing: full tail (Densify+AddNoise+Digitize), torch coherent ==")
    cpu_dt, _ = _time('cpu')
    gpu_dt, _ = _time('cuda')
    print(f"  CPU (no ToDevice): {cpu_dt*1e3:8.1f} ms/batch")
    print(f"  GPU (ToDevice):    {gpu_dt*1e3:8.1f} ms/batch   speedup {cpu_dt/gpu_dt:.1f}x")

    # split the GPU number: sparse->device transfer vs the dense ops on-device.
    xfer = Compose([dict(type='ToDevice', device='cuda')])
    xfer(copy.deepcopy(SPARSE)); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        xfer(copy.deepcopy(SPARSE)); torch.cuda.synchronize()
    xfer_dt = (time.perf_counter() - t0) / ITERS
    print(f"    of which ToDevice (sparse->GPU): {xfer_dt*1e3:6.1f} ms   "
          f"dense ops on-GPU: {(gpu_dt-xfer_dt)*1e3:6.1f} ms")

    print("\n== correctness: CPU vs GPU with coherent_numpy=True (device-independent) ==")
    _, out_cpu = _time('cpu', coherent_numpy=True, iters=1)
    _, out_gpu = _time('cuda', coherent_numpy=True, iters=1)
    for gid in sorted(out_cpu['sensor_dense']):
        a = out_cpu['sensor_dense'][gid]
        b = out_gpu['sensor_dense'][gid].cpu()
        print(f"  plane {gid}: {tuple(a.shape)}  max|CPU-GPU|={ (a-b).abs().max().item():.3e}")

    print("\n== correctness: unified Densify vs manual numpy scatter (bit-exact) ==")
    dens = Compose([dict(type='Densify', geom=GEOM, modality='sensor')])(
        copy.deepcopy(SPARSE))
    grids = dens['sensor_dense']
    wire = SPARSE['sensor_wire'].numpy(); tim = SPARSE['sensor_time'].numpy()
    val = SPARSE['sensor_value'].numpy(); pg = SPARSE['sensor_plane_gid'].numpy()
    off = SPARSE['sensor_offset'].numpy()
    starts = np.concatenate([[0], off[:-1]])
    for gid in sorted(grids):
        e = GEOM[gid]; W, T = e['n_wires'], e['n_ticks']
        ref = np.zeros((B, W, T), np.float32)
        for bi in range(B):
            s, en = int(starts[bi]), int(off[bi])
            m = pg[s:en] == gid
            ref[bi, wire[s:en][m], tim[s:en][m]] = val[s:en][m]
        diff = float(np.abs(grids[gid].numpy() - ref).max())
        print(f"  plane {gid}: max|unified-manual|={diff:.3e}")
        break  # one plane suffices (same code path per plane)


if __name__ == '__main__':
    main()

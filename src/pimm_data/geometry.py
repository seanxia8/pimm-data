"""Load a JAXTPC-exported plane-geometry JSON into the registry the dense sensor
path consumes. The JSON is produced by ``JAXTPC/scripts/export_plane_geometry.py``
from the detector config (the geometry source of truth); pimm-data reads the plain
JSON — **no JAXTPC dependency**.

    geom = load_plane_registry("cubic_wireplane_geometry.json")   # ships in data/
    stages = build_sensor_gpu_stages(geom, coherent=True, incoherent=True)

The registry is keyed by canonical plane id (matching the per-point ``plane_gid``
the sensor reader surfaces), with per-plane ``n_wires``/``n_ticks``/``pedestal``/
``wire_lengths`` (meters) — exactly what densify / add_intrinsic_noise / digitize need.
"""

import json
import os

import numpy as np

from .jaxtpc import canonical_plane_id

_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def _resolve(path):
    for cand in (path, os.path.join(_DATA_DIR, path),
                 os.path.join(_DATA_DIR, str(path) + '.json')):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"plane-geometry file {path!r} not found (looked in cwd and {_DATA_DIR})")


def _wire_lengths(spec, n_wires):
    """Constant planes store a scalar (collection); varying ones a per-wire array."""
    if isinstance(spec, (int, float)):
        return np.full(int(n_wires), float(spec), dtype=np.float32)
    arr = np.asarray(spec, dtype=np.float32)
    if arr.shape != (int(n_wires),):
        raise ValueError(f"wire_lengths length {arr.shape} != n_wires {n_wires}")
    return arr


def load_plane_registry(path):
    """JSON -> ``{canonical_plane_id(label): {label, n_wires, n_ticks, pedestal, wire_lengths}}``."""
    with open(_resolve(path)) as f:
        d = json.load(f)
    nts = int(d['num_time_steps'])
    reg = {}
    for label, e in d['planes'].items():
        nw = int(e['n_wires'])
        reg[canonical_plane_id(label)] = {
            'label': label, 'n_wires': nw, 'n_ticks': nts,
            'pedestal': int(e['pedestal']),
            'wire_lengths': _wire_lengths(e['wire_lengths_m'], nw),
        }
    return reg


def dataset_geometry_kwargs(path):
    """JSON -> kwargs for ``JAXTPCDataset`` (num_time_steps + per-plane geometry),
    so the dataset can surface a fixed grid even when the files lack the attrs."""
    with open(_resolve(path)) as f:
        d = json.load(f)
    nwbp, pbp, wlbp = {}, {}, {}
    for label, e in d['planes'].items():
        nw = int(e['n_wires'])
        nwbp[label] = nw
        pbp[label] = int(e['pedestal'])
        wlbp[label] = _wire_lengths(e['wire_lengths_m'], nw)  # full array
    return dict(num_time_steps=int(d['num_time_steps']), n_wires_per_plane=nwbp,
                pedestal_per_plane=pbp, wire_lengths_per_plane=wlbp)

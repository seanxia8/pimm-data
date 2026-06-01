"""
LUCiDHitsReader — per-particle PMT hit decomposition from LUCiD ``hits/``
HDF5 files (``format_version: 3``).

Each row is a ``(sensor_idx, particle_idx)`` entry: the contribution of one
particle to one PMT. The same ``sensor_idx`` appears multiple times when
several particles illuminate the same PMT. ``particle_idx`` indexes into
the per-event particle table (see :class:`LUCiDLablReader`).

Output dict (flat):

    sensor_idx       (E,) int32
    particle_idx     (E,) int32
    pe               (E,) float32
    t                (E,) float32
"""

import numpy as np

from ._base import ShardReaderBase


class LUCiDHitsReader(ShardReaderBase):
    """Reads per-particle hit decomposition from LUCiD ``hits/`` files.

    Parameters
    ----------
    data_root : str
        Directory containing hits shard files.
    split : str
        Split name (used as subdirectory when present).
    dataset_name : str
        File prefix — matches ``{dataset_name}_hits_*.h5``.
    pe_threshold : float
        If > 0, drop entries with ``pe <= pe_threshold``.
    """

    _MODALITY = 'hits'

    def __init__(self, data_root, split='', dataset_name='wc',
                 pe_threshold=0.0, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.pe_threshold = float(pe_threshold)
        self._init_shards()

    def read_event(self, idx):
        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        sensor_idx = evt['sensor_idx'][:].astype(np.int32)
        particle_idx = evt['particle_idx'][:].astype(np.int32)
        pe = evt['PE'][:].astype(np.float32)
        t = evt['T'][:].astype(np.float32)

        if self.pe_threshold > 0 and pe.size > 0:
            mask = pe > self.pe_threshold
            sensor_idx = sensor_idx[mask]
            particle_idx = particle_idx[mask]
            pe = pe[mask]
            t = t[mask]

        return {
            'sensor_idx': sensor_idx,
            'particle_idx': particle_idx,
            'pe': pe,
            't': t,
        }

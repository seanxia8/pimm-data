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

import os
import glob
import logging
import numpy as np
import h5py

from .._shard_meta import read_shard_meta

log = logging.getLogger(__name__)


class LUCiDHitsReader:
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

    def __init__(self, data_root, split='', dataset_name='wc',
                 pe_threshold=0.0, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.pe_threshold = float(pe_threshold)

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No LUCiD hits files found for '{dataset_name}' in "
            f"{data_root}/{split}")

        self._initted = False
        self._h5data = []
        self._build_index()

    def _find_files(self):
        for pattern in (
            os.path.join(self.data_root, self.split,
                         f'{self.dataset_name}_hits_*.h5'),
            os.path.join(self.data_root, f'{self.dataset_name}_hits_*.h5'),
        ):
            files = sorted(glob.glob(pattern))
            if files:
                return files
        return []

    def _build_index(self):
        self.cumulative_lengths = []
        self.indices = []

        for h5_path in self.h5_files:
            try:
                index = read_shard_meta(h5_path)['present_events']
            except Exception as e:
                log.warning("Error processing %s: %s", h5_path, e)
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info("LUCiDHitsReader: %d events from %d files",
                 self.cumulative_lengths[-1], len(self.h5_files))

    def h5py_worker_init(self):
        self._h5data = [
            h5py.File(p, 'r', libver='latest', swmr=True)
            for p in self.h5_files
        ]
        self._initted = True

    def _locate_event(self, idx):
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx,
                                       side='right'))
        local_idx = idx - (int(self.cumulative_lengths[file_idx - 1])
                           if file_idx > 0 else 0)
        event_num = self.indices[file_idx][local_idx]
        event_key = f'event_{event_num:03d}'
        return self._h5data[file_idx], event_key

    def read_event(self, idx):
        if not self._initted:
            self.h5py_worker_init()

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

    def __len__(self):
        return (int(self.cumulative_lengths[-1])
                if len(self.cumulative_lengths) > 0 else 0)

    def close(self):
        if self._initted:
            for fh in self._h5data:
                try:
                    fh.close()
                except Exception:
                    pass
            self._h5data = []
            self._initted = False

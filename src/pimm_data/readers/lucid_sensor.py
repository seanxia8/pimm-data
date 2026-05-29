"""
LUCiDSensorReader — reads sparse PMT hits from LUCiD ``sensor/`` HDF5
files (``format_version: 3``).

Layout: per-event groups. Each ``event_XXX`` holds flat per-hit arrays
``PE``, ``T``, ``sensor_idx``. The full PMT position table lives once per
file at ``config/sensor_positions``. Per-particle decomposition and
label columns are *not* in this file — they live in ``hits/`` and
``labl/``.

Output dict:

    sensor_idx  (H,)          int32    — PMT index per hit
    pmt_pe      (H,)          float32
    pmt_t       (H,)          float32
    pmt_coord   (n_sensors,3) float32  — full PMT geometry
"""

import os
import glob
import logging
import numpy as np
import h5py

from .._shard_meta import read_shard_meta, open_event_files

log = logging.getLogger(__name__)


class LUCiDSensorReader:
    """Reads sparse per-event PMT hits from LUCiD ``sensor/`` files.

    Parameters
    ----------
    data_root : str
        Directory containing sensor shard files.
    split : str
        Split name (used as subdirectory when present).
    dataset_name : str
        File prefix — matches ``{dataset_name}_sensor_*.h5``.
    pmt_positions : ndarray or None
        Optional override for ``config/sensor_positions``. Kept for
        geometry substitution experiments; normally leave ``None``.
    pmt_positions_file : str or None
        Optional path to a ``.npy`` file overriding PMT positions.
    """

    def __init__(self, data_root, split='', dataset_name='wc',
                 pmt_positions=None, pmt_positions_file=None, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name

        if pmt_positions is not None:
            self._pmt_positions = np.asarray(pmt_positions, dtype=np.float32)
        elif pmt_positions_file is not None:
            self._pmt_positions = np.load(pmt_positions_file).astype(np.float32)
        else:
            self._pmt_positions = None

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No LUCiD sensor files found for '{dataset_name}' in "
            f"{data_root}/{split}")

        self._initted = False
        self._h5data = []
        self._n_sensors = None
        self._build_index()

    def _find_files(self):
        for pattern in (
            os.path.join(self.data_root, self.split,
                         f'{self.dataset_name}_sensor_*.h5'),
            os.path.join(self.data_root, f'{self.dataset_name}_sensor_*.h5'),
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
                meta = read_shard_meta(h5_path)
                if self._n_sensors is None:
                    self._n_sensors = int(meta['config_attrs'].get('n_sensors', 0))
                index = meta['present_events']
            except Exception as e:
                log.warning("Error processing %s: %s", h5_path, e)
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info("LUCiDSensorReader: %d events, %d sensors from %d files",
                 self.cumulative_lengths[-1], self._n_sensors or 0,
                 len(self.h5_files))

    def h5py_worker_init(self):
        self._h5data = open_event_files(self.h5_files, self.indices)
        if self._pmt_positions is None:
            # First shard that actually opened (file 0 may be a skipped
            # empty/dangling shard — F17). PMT geometry is shared across shards.
            f0 = next((h for h in self._h5data if h is not None), None)
            if f0 is not None and 'sensor_positions' in f0['config']:
                self._pmt_positions = f0['config']['sensor_positions'][:].astype(
                    np.float32)
        self._initted = True

    def _locate_event(self, idx):
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx,
                                       side='right'))
        local_idx = idx - (int(self.cumulative_lengths[file_idx - 1])
                           if file_idx > 0 else 0)
        event_num = self.indices[file_idx][local_idx]
        return self._h5data[file_idx], f'event_{event_num:03d}'

    def read_event(self, idx):
        if not self._initted:
            self.h5py_worker_init()

        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        data = {
            'sensor_idx': evt['sensor_idx'][:].astype(np.int32),
            'pmt_pe':     evt['PE'][:].astype(np.float32),
            'pmt_t':      evt['T'][:].astype(np.float32),
        }
        if self._pmt_positions is not None:
            data['pmt_coord'] = self._pmt_positions
        return data

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

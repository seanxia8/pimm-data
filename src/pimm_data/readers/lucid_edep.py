"""
LUCiDEdepReader — 3D segment deposits from LUCiD ``edep/`` HDF5 files
(``format_version: 3``).

Each ``event_XXX`` holds per-segment arrays. One "segment" is one
Geant4 step; many segments per track, many tracks per particle
(``particle_idx`` = FK into :class:`LUCiDLablReader`'s ``per_particle``;
``track_idx`` = FK into ``per_track``).

Output dict:

    coord       (N,3) float32   midpoint of start/end
    energy      (N,1) float32   edep
    time        (N,1) float32
    track_idx   (N,)  int32     FK → labl.per_track
    contained   (N,)  bool      step start+end both inside detector volume

    (include_physics=True adds:)
    direction   (N,3) float32   unit direction
    beta_start  (N,1) float32   initial beta (v/c) at segment start
    n_cherenkov (N,1) int32     number of Cherenkov photons produced
"""

import os
import glob
import logging
import numpy as np
import h5py

from .._shard_meta import read_shard_meta, open_event_files

log = logging.getLogger(__name__)


class LUCiDEdepReader:
    """Reads 3D segment deposits from LUCiD ``edep/`` files.

    Parameters
    ----------
    data_root : str
        Directory containing edep shard files.
    split : str
        Split name (used as subdirectory when present).
    dataset_name : str
        File prefix — matches ``{dataset_name}_edep_*.h5``.
    min_segments : int
        Drop events with fewer than this many segments.
    include_physics : bool
        Also emit ``direction``, ``beta_start``, ``n_cherenkov``.
    """

    def __init__(self, data_root, split='', dataset_name='wc',
                 min_segments=0, include_physics=True, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.min_segments = int(min_segments)
        self.include_physics = bool(include_physics)

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No LUCiD edep files found for '{dataset_name}' in "
            f"{data_root}/{split}")

        self._initted = False
        self._h5data = []
        self._build_index()

    def _find_files(self):
        for pattern in (
            os.path.join(self.data_root, self.split,
                         f'{self.dataset_name}_edep_*.h5'),
            os.path.join(self.data_root, f'{self.dataset_name}_edep_*.h5'),
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
                if self.min_segments > 0:
                    with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                        present = read_shard_meta(h5_path)['present_events']
                        valid = []
                        for i in present:
                            ek = f'event_{int(i):03d}'
                            if ek not in f:
                                continue
                            n_seg = int(f[ek].attrs.get('n_segments', 0))
                            if n_seg >= self.min_segments:
                                valid.append(int(i))
                        index = np.array(valid, dtype=np.int64)
                else:
                    index = read_shard_meta(h5_path)['present_events']
            except Exception as e:
                log.warning("Error processing %s: %s", h5_path, e)
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info("LUCiDEdepReader: %d events from %d files (min_segments=%d)",
                 self.cumulative_lengths[-1], len(self.h5_files),
                 self.min_segments)

    def h5py_worker_init(self):
        self._h5data = open_event_files(self.h5_files, self.indices)
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

        sx = evt['start_x'][:].astype(np.float32)
        sy = evt['start_y'][:].astype(np.float32)
        sz = evt['start_z'][:].astype(np.float32)
        ex = evt['end_x'][:].astype(np.float32)
        ey = evt['end_y'][:].astype(np.float32)
        ez = evt['end_z'][:].astype(np.float32)

        coord = np.stack([(sx + ex) * 0.5,
                          (sy + ey) * 0.5,
                          (sz + ez) * 0.5], axis=1)

        data = {
            'coord': coord,
            'energy': evt['edep'][:].astype(np.float32)[:, None],
            'time': evt['time'][:].astype(np.float32)[:, None],
            'track_idx': evt['track_idx'][:].astype(np.int32),
        }
        if 'contained' in evt:
            data['contained'] = evt['contained'][:].astype(bool)

        if self.include_physics:
            direction = np.stack([evt['dir_x'][:].astype(np.float32),
                                  evt['dir_y'][:].astype(np.float32),
                                  evt['dir_z'][:].astype(np.float32)],
                                 axis=1)
            data['direction'] = direction
            data['beta_start'] = evt['beta_start'][:].astype(
                np.float32)[:, None]
            data['n_cherenkov'] = evt['n_cherenkov'][:].astype(
                np.int32)[:, None]

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

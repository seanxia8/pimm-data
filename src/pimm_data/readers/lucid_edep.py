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

import numpy as np
import h5py

from ._base import ShardReaderBase


class LUCiDEdepReader(ShardReaderBase):
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

    _MODALITY = 'edep'

    def __init__(self, data_root, split='', dataset_name='wc',
                 min_segments=0, include_physics=True, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.min_segments = int(min_segments)
        self.include_physics = bool(include_physics)
        self._init_shards()

    def _index_for_shard(self, h5_path):
        """Present events, filtered by ``min_segments`` when set.

        One open: iterate the present ``event_*`` groups directly and keep
        those with enough segments (gap-tolerant — only real groups visited).
        With the filter off, falls back to the base's present-events."""
        if self.min_segments > 0:
            with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                valid = [
                    int(k.rsplit('_', 1)[1]) for k in f.keys()
                    if k.startswith('event_')
                    and int(f[k].attrs.get('n_segments', 0))
                    >= self.min_segments]
            return np.array(sorted(valid), dtype=np.int64)
        return super()._index_for_shard(h5_path)

    def read_event(self, idx):
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

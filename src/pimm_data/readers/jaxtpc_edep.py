"""
JAXTPCEdepReader — reads 3D truth energy deposits from JAXTPC edep files.

Produces raw geometry and physics as numpy arrays. Labels are not
applied here — that is the dataset's responsibility (see jaxtpc.py).

Output dict:
    coord (N,3), energy (N,1), volume_id (N,1),
    and optionally: dx, theta, phi, t0_us, charge, photons
"""

import os
import glob
import logging
import numpy as np
import h5py

from .._shard_meta import read_shard_meta, read_deposit_counts

log = logging.getLogger(__name__)


class JAXTPCEdepReader:
    """Reads 3D truth deposits from JAXTPC edep HDF5 files.

    Concatenates volumes into a single point cloud with a volume_id feature.
    No label computation — just raw data.

    Parameters
    ----------
    data_root : str
        Directory containing edep shard files.
    split : str
        Split name — used as subdirectory or glob pattern.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_edep_0000.h5').
    min_deposits : int
        Minimum deposits per event to include in index.
    include_physics : bool
        Whether to load dx, theta, phi, charge, photons, etc.
    volume : int or None
        Load only this volume index. None = all volumes.
    """

    def __init__(self, data_root, split='train', dataset_name='sim',
                 min_deposits=0, include_physics=True, volume=None):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.min_deposits = min_deposits
        self.include_physics = include_physics
        self.volume = volume

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No edep files found for '{dataset_name}' in {data_root}/{split}")

        self._initted = False
        self._h5data = []

        self._build_index()

    def _find_files(self):
        """Discover edep shard files."""
        pattern = os.path.join(
            self.data_root, self.split,
            f'{self.dataset_name}_edep_*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(
                self.data_root, f'{self.dataset_name}_edep_*.h5')
            files = sorted(glob.glob(pattern))
        return files

    def _build_index(self):
        """Scan files, count events, build cumulative index."""
        self.cumulative_lengths = []
        self.indices = []

        for h5_path in self.h5_files:
            try:
                if self.min_deposits > 0:
                    # Per-event deposit counts via the cached scan (F16) — the
                    # dominant index-build cost when min_deposits>0; memoized so
                    # train/val/test (and tiered) datasets over the same shards
                    # share one pass. Counts are over present events only.
                    counts = read_deposit_counts(h5_path)
                    valid = [num for num in sorted(counts)
                             if self._count_from_cache(counts[num])
                             >= self.min_deposits]
                    index = np.array(valid, dtype=np.int64)
                else:
                    # Index from event groups actually present, not
                    # arange(n_events): production may skip an event
                    # (capacity overflow) leaving a gap — arange would
                    # KeyError at read time. (Cached scan — A1.)
                    index = read_shard_meta(h5_path)['present_events']

            except Exception as e:
                log.warning("Error processing %s: %s", h5_path, e)
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info("JAXTPCEdepReader: %d events from %d files (min_deposits=%d)",
                 self.cumulative_lengths[-1], len(self.h5_files),
                 self.min_deposits)

    def _count_from_cache(self, c):
        """Deposit total for the ``min_deposits`` filter from a cached
        :func:`read_deposit_counts` entry.

        Volume-aware (A3): when ``self.volume`` is set, count only that
        volume's deposits — i.e. exactly what ``read_event`` will return — so
        an event whose deposits all live in *another* volume is excluded
        rather than kept and then read back empty. With no volume filter the
        count is the sum over all present volumes.
        """
        if c['n_volumes'] > 1:
            if self.volume is not None:
                pv = c['per_vol']
                return pv[self.volume] if self.volume < len(pv) else 0
            return sum(c['per_vol'])
        return c['positions']

    def h5py_worker_init(self):
        """Lazily open file handles (called after DataLoader fork)."""
        self._h5data = [
            h5py.File(p, 'r', libver='latest', swmr=True)
            for p in self.h5_files
        ]
        self._initted = True

    def _locate_event(self, idx):
        """Map global index → (file_handle, event_key, n_volumes)."""
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side='right'))
        local_idx = idx - (int(self.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0)
        event_num = self.indices[file_idx][local_idx]
        event_key = f'event_{event_num:03d}'
        f = self._h5data[file_idx]
        n_volumes = int(f['config'].attrs.get('n_volumes', 1))
        return f, event_key, n_volumes

    def read_event(self, idx):
        """Read one event, return flat dict of numpy arrays.

        No label computation — just raw geometry, physics, and IDs.
        """
        if not self._initted:
            self.h5py_worker_init()

        f, event_key, n_volumes = self._locate_event(idx)
        evt = f[event_key]

        vol_arrays = []

        if n_volumes > 1:
            for v in range(n_volumes):
                if self.volume is not None and v != self.volume:
                    continue
                vk = f'volume_{v}'
                if vk not in evt:
                    continue
                vg = evt[vk]
                n = int(vg.attrs.get('n_actual', 0))
                if n == 0:
                    continue
                vol_arrays.append(self._read_volume(vg, n, v))
        else:
            if 'positions' in evt:
                n = evt['positions'].shape[0]
                vol_arrays.append(self._read_volume_flat(evt, n, 0))

        if not vol_arrays:
            return self._empty_dict()

        return self._concat_volumes(vol_arrays)

    def _read_volume(self, vg, n, vol_idx):
        """Read physics arrays from a volume group.

        Edep carries only deposit-level physics. Instance identifiers
        (group_ids, deposit→group FK) live in hits; per-track metadata
        (pdg, interaction, ancestor) lives in labl.
        """
        step = float(vg.attrs['pos_step_mm'])
        origin = np.array([vg.attrs['pos_origin_x'],
                           vg.attrs['pos_origin_y'],
                           vg.attrs['pos_origin_z']], dtype=np.float32)

        pos = vg['positions'][:].astype(np.float32)
        pos *= step       # in-place: avoid the *step and +origin temporaries
        pos += origin
        d = {
            'coord': pos,
            'energy': vg['de'][:].astype(np.float32),
            'volume_id': np.full(n, vol_idx, dtype=np.int32),
        }

        if self.include_physics:
            for key in ('dx', 'theta', 'phi', 't0_us'):
                if key in vg:
                    d[key] = vg[key][:].astype(np.float32)
            for key in ('charge', 'photons'):
                if key in vg:
                    d[key] = vg[key][:].astype(np.float32)

        return d

    def _read_volume_flat(self, evt, n, vol_idx):
        """Read from legacy flat event format (no volume subgroups)."""
        step = float(evt.attrs['pos_step_mm'])
        origin = np.array([evt.attrs['pos_origin_x'],
                           evt.attrs['pos_origin_y'],
                           evt.attrs['pos_origin_z']], dtype=np.float32)

        pos = evt['positions'][:].astype(np.float32)
        pos *= step
        pos += origin
        d = {
            'coord': pos,
            'energy': evt['de'][:].astype(np.float32),
            'volume_id': np.full(n, vol_idx, dtype=np.int32),
        }

        if self.include_physics:
            for key in ('dx', 'theta', 'phi', 't0_us'):
                if key in evt:
                    d[key] = evt[key][:].astype(np.float32)
            for key in ('charge', 'photons'):
                if key in evt:
                    d[key] = evt[key][:].astype(np.float32)

        return d

    def _concat_volumes(self, vol_arrays):
        """Concatenate per-volume dicts into a single flat dict."""
        keys = vol_arrays[0].keys()
        data_dict = {}
        for k in keys:
            arrays = [v[k] for v in vol_arrays if k in v]
            # Single-volume (common with volume= filter): skip the
            # concatenate, which would otherwise copy every array.
            combined = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
            if k == 'coord':
                data_dict[k] = combined
            elif k in ('energy', 'dx', 'theta', 'phi', 't0_us',
                       'charge', 'photons'):
                data_dict[k] = combined[:, None]
            elif k == 'volume_id':
                data_dict[k] = combined[:, None]
            else:
                data_dict[k] = combined

        return data_dict

    def _empty_dict(self):
        """Minimal valid dict for empty events."""
        return {
            'coord': np.zeros((0, 3), dtype=np.float32),
            'energy': np.zeros((0, 1), dtype=np.float32),
            'volume_id': np.zeros((0, 1), dtype=np.int32),
        }

    def __len__(self):
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) > 0 else 0

    def close(self):
        """Close open file handles."""
        if self._initted:
            for f in self._h5data:
                try:
                    f.close()
                except Exception:
                    pass
            self._h5data = []
            self._initted = False

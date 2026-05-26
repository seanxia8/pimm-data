"""
JAXTPCHitsReader — reads per-particle charge attribution from JAXTPC
``hits/`` files. Supports both wire and pixel readouts.

Readout type is auto-detected from ``/config.readout_type`` and exposed
as ``reader.readout_type``. Decoding of CSR-encoded per-plane
correspondence yields:

- wire:  ``hits.{plane}.{wire, time, group_id, charge}``
- pixel: ``hits.{plane}.{py, pz, time, group_id, charge}``

Also loads per-volume ``group_to_track``, ``deposit_to_group``, and
``qs_fractions`` lookup tables (same shape for both readouts).

All decoding is fully vectorized (no Python loops over groups).
"""

import os
import glob
import logging
import numpy as np
import h5py

log = logging.getLogger(__name__)


class JAXTPCHitsReader:
    """Reads per-particle charge attribution from JAXTPC ``hits/`` HDF5 files.

    Parameters
    ----------
    data_root : str
        Directory containing hits shard files.
    split : str
        Split name.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_hits_0000.h5').
    planes : str or list
        Which planes to load: 'all' or list like ['volume_0_U'].
    """

    def __init__(self, data_root, split='train', dataset_name='sim',
                 planes='all', **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.planes = planes

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No hits files found for '{dataset_name}' in {data_root}/{split}")

        self._initted = False
        self._h5data = []

        self._build_index()
        self.readout_type = self._detect_readout_type()

    def _find_files(self):
        """Locate hits shard files."""
        pattern = os.path.join(
            self.data_root, self.split,
            f'{self.dataset_name}_hits_*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(
                self.data_root, f'{self.dataset_name}_hits_*.h5')
            files = sorted(glob.glob(pattern))
        return files

    def _build_index(self):
        self.cumulative_lengths = []
        self.indices = []

        for h5_path in self.h5_files:
            try:
                with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                    # Index from event groups actually present, not
                    # arange(n_events): production may skip an event (e.g.
                    # capacity overflow), leaving a gap with n_events
                    # unchanged — arange would then KeyError at read time.
                    index = np.array(sorted(
                        int(k.rsplit('_', 1)[1]) for k in f.keys()
                        if k.startswith('event_')), dtype=np.int64)
            except Exception as e:
                log.warning("Error processing %s: %s", h5_path, e)
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info("JAXTPCHitsReader: %d events from %d files",
                 self.cumulative_lengths[-1], len(self.h5_files))

    def h5py_worker_init(self):
        self._h5data = [
            h5py.File(p, 'r', libver='latest', swmr=True)
            for p in self.h5_files
        ]
        self._initted = True

    def _locate_event(self, idx):
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side='right'))
        local_idx = idx - (int(self.cumulative_lengths[file_idx - 1]) if file_idx > 0 else 0)
        event_num = self.indices[file_idx][local_idx]
        event_key = f'event_{event_num:03d}'
        f = self._h5data[file_idx]
        return f, event_key

    def _detect_readout_type(self):
        """Return 'pixel' or 'wire' based on the first file's config attr.

        Falls back to inspecting plane datasets if the attr is absent.
        """
        for path in self.h5_files:
            try:
                with h5py.File(path, 'r', libver='latest', swmr=True) as f:
                    if 'config' in f and 'readout_type' in f['config'].attrs:
                        rt = str(f['config'].attrs['readout_type'])
                        if rt in ('wire', 'pixel'):
                            return rt
                    for ek in f:
                        if not ek.startswith('event_'):
                            continue
                        evt = f[ek]
                        for vk in evt:
                            vol = evt[vk]
                            if not isinstance(vol, h5py.Group) or not vk.startswith('volume_'):
                                continue
                            for pk in vol:
                                pg = vol[pk]
                                if not isinstance(pg, h5py.Group):
                                    continue
                                if 'center_py' in pg or 'delta_py' in pg:
                                    return 'pixel'
                                if 'center_wires' in pg or 'delta_wires' in pg:
                                    return 'wire'
                        break
            except Exception as e:
                log.warning("readout detection failed on %s: %s", path, e)
                continue
        return 'wire'

    @staticmethod
    def _decode_plane_wire(g):
        """Decode one wire plane's CSR correspondence — fully vectorized.

        Returns (wire, time, group_id, charge) arrays, all shape (E,).
        """
        group_ids = g['group_ids'][:]
        group_sizes = g['group_sizes'][:].astype(np.int32)
        center_wires = g['center_wires'][:]
        center_times = g['center_times'][:]
        peak_charges = g['peak_charges'][:]
        delta_wires = g['delta_wires'][:]
        delta_times = g['delta_times'][:]
        charges_u16 = g['charges_u16'][:]

        G = len(group_ids)
        if G == 0:
            empty = np.array([], dtype=np.int32)
            return empty, empty, empty, np.array([], dtype=np.float32)

        # In-place ops avoid full-size int32/float32 temporaries (the
        # explicit .astype + separate add/mul/div allocated several copies
        # of millions of entries per plane).
        wires = np.repeat(center_wires, group_sizes).astype(np.int32)
        wires += delta_wires
        times = np.repeat(center_times, group_sizes).astype(np.int32)
        times += delta_times
        gids = np.repeat(group_ids, group_sizes)
        charges = np.repeat(peak_charges, group_sizes)  # float32
        charges *= charges_u16
        charges *= np.float32(1.0 / 65535.0)

        return wires, times, gids, charges

    @staticmethod
    def _decode_plane_pixel(g):
        """Decode one pixel plane's CSR correspondence — fully vectorized.

        Returns (py, pz, time, group_id, charge) arrays, all shape (E,).
        """
        group_ids = g['group_ids'][:]
        group_sizes = g['group_sizes'][:].astype(np.int32)
        center_py = g['center_py'][:]
        center_pz = g['center_pz'][:]
        center_times = g['center_times'][:]
        peak_charges = g['peak_charges'][:]
        delta_py = g['delta_py'][:]
        delta_pz = g['delta_pz'][:]
        delta_times = g['delta_times'][:]
        # Pixel writer emits signed charges_i16 normalized by 32767 (see
        # encode_correspondence_csr_pixel); wire uses charges_u16 / 65535.
        charges_i16 = g['charges_i16'][:]

        G = len(group_ids)
        if G == 0:
            empty = np.array([], dtype=np.int32)
            return empty, empty, empty, empty, np.array([], dtype=np.float32)

        # In-place ops avoid full-size int32/float32 temporaries.
        py = np.repeat(center_py, group_sizes).astype(np.int32)
        py += delta_py
        pz = np.repeat(center_pz, group_sizes).astype(np.int32)
        pz += delta_pz
        times = np.repeat(center_times, group_sizes).astype(np.int32)
        times += delta_times
        gids = np.repeat(group_ids, group_sizes)
        # charges_i16 already carries each entry's sign (the writer normalizes
        # by abs(peak), so charge = |peak| * i16/32767). Multiplying by the
        # *signed* peak would flip every charge in a negative-peak group.
        charges = np.abs(np.repeat(peak_charges, group_sizes))  # float32
        charges *= charges_i16
        charges *= np.float32(1.0 / 32767.0)

        return py, pz, times, gids, charges

    def read_event(self, idx):
        """Read one event's per-particle charge attribution.

        Wire returns:
            hits.{vol_plane}.{wire, time, group_id, charge}
        Pixel returns:
            hits.{vol_plane}.{py, pz, time, group_id, charge}

        Plus (both readouts):
            group_to_track_v{N}, deposit_to_group_v{N}, qs_fractions_v{N}
        """
        if not self._initted:
            self.h5py_worker_init()

        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        data_dict = {}

        for vol_key in evt:
            vol = evt[vol_key]
            if not isinstance(vol, h5py.Group):
                continue
            if not vol_key.startswith('volume_'):
                continue

            vol_idx = vol_key.replace('volume_', '')

            # Per-volume (per-deposit and per-group) arrays
            if 'group_to_track' in vol:
                data_dict[f'group_to_track_v{vol_idx}'] = \
                    vol['group_to_track'][:].astype(np.int32)
            if 'deposit_to_group' in vol:
                data_dict[f'deposit_to_group_v{vol_idx}'] = \
                    vol['deposit_to_group'][:].astype(np.int32)
            if 'qs_fractions' in vol:
                data_dict[f'qs_fractions_v{vol_idx}'] = \
                    vol['qs_fractions'][:].astype(np.float32)

            # Per-plane CSR-decoded pixel entries
            for plane_key in vol:
                pg = vol[plane_key]
                if not isinstance(pg, h5py.Group) or 'group_ids' not in pg:
                    continue

                plane_label = f'volume_{vol_idx}_{plane_key}'
                if self.planes != 'all' and plane_label not in self.planes:
                    continue

                prefix = f'hits.{plane_label}'
                if self.readout_type == 'pixel':
                    py, pz, times, gids, charges = self._decode_plane_pixel(pg)
                    data_dict[f'{prefix}.py'] = py
                    data_dict[f'{prefix}.pz'] = pz
                    data_dict[f'{prefix}.time'] = times
                    data_dict[f'{prefix}.group_id'] = gids
                    data_dict[f'{prefix}.charge'] = charges
                else:
                    wires, times, gids, charges = self._decode_plane_wire(pg)
                    data_dict[f'{prefix}.wire'] = wires
                    data_dict[f'{prefix}.time'] = times
                    data_dict[f'{prefix}.group_id'] = gids
                    data_dict[f'{prefix}.charge'] = charges

        return data_dict

    def __len__(self):
        return int(self.cumulative_lengths[-1]) if len(self.cumulative_lengths) > 0 else 0

    def close(self):
        if self._initted:
            for f in self._h5data:
                try:
                    f.close()
                except Exception:
                    pass
            self._h5data = []
            self._initted = False

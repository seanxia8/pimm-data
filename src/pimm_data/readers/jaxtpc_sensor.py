"""
JAXTPCSensorReader — reads sparse raw sensor signals from JAXTPC
``sensor/`` files. Supports both wire and pixel readouts.

Readout type is auto-detected from ``/config.readout_type`` and exposed
as ``reader.readout_type`` (``'wire'`` or ``'pixel'``). Output schema:

- wire:  ``sensor.{plane_label}.{wire, time, value}``
- pixel: ``sensor.{plane_label}.{py, pz, time, value}``

Handles both old format (planes directly under event) and new format
(planes under volume_N/ subgroups).
"""

import os
import glob
import logging
import numpy as np
import h5py

from .._shard_meta import read_shard_meta

log = logging.getLogger(__name__)


class JAXTPCSensorReader:
    """Reads sparse raw sensor signals from JAXTPC ``sensor/`` HDF5 files.

    Parameters
    ----------
    data_root : str
        Directory containing sensor shard files.
    split : str
        Split name — used as subdirectory or glob pattern.
    dataset_name : str
        File prefix (e.g., 'sim' matches 'sim_sensor_0000.h5').
    planes : str or list
        Which planes to load: 'all' or list like ['east_U', 'east_V'].
    decode_digitization : bool
        If True, subtract pedestal from uint16 values.
    """

    def __init__(self, data_root, split='train', dataset_name='sim',
                 planes='all', decode_digitization=True):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.planes = planes
        self.decode_digitization = decode_digitization

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No sensor files found for '{dataset_name}' in {data_root}/{split}")

        self._initted = False
        self._h5data = []

        self._build_index()
        self.readout_type = self._detect_readout_type()

    def _find_files(self):
        """Locate sensor shard files."""
        pattern = os.path.join(
            self.data_root, self.split,
            f'{self.dataset_name}_sensor_*.h5')
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(
                self.data_root, f'{self.dataset_name}_sensor_*.h5')
            files = sorted(glob.glob(pattern))
        return files

    def _build_index(self):
        self.cumulative_lengths = []
        self.indices = []

        for h5_path in self.h5_files:
            try:
                # Index from event groups actually present, not
                # arange(n_events): production may skip an event (e.g.
                # capacity overflow), leaving a gap with n_events unchanged —
                # arange would then KeyError at read time. (Cached scan; the
                # readout-type probe reuses the same open — A1.)
                index = read_shard_meta(h5_path)['present_events']
            except Exception as e:
                log.warning("Error processing %s: %s", h5_path, e)
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info("JAXTPCSensorReader: %d events from %d files",
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

        Falls back to inspecting plane datasets if the attr is absent
        (older files written before the attr was added).
        """
        for path in self.h5_files:
            try:
                # Common case: the readout_type config attr is present — the
                # cached meta (already read in _build_index) answers without
                # a second file open (A1).
                rt = read_shard_meta(path)['config_attrs'].get('readout_type')
                if rt is not None:
                    rt = str(rt)
                    if rt in ('wire', 'pixel'):
                        return rt
                # Attr absent (older files) → inspect plane datasets.
                with h5py.File(path, 'r', libver='latest', swmr=True) as f:
                    for ek in f:
                        if not ek.startswith('event_'):
                            continue
                        evt = f[ek]
                        for vk in evt:
                            vol = evt[vk]
                            if not isinstance(vol, h5py.Group):
                                continue
                            if vk.startswith('volume_'):
                                for pk in vol:
                                    pg = vol[pk]
                                    if not isinstance(pg, h5py.Group):
                                        continue
                                    if 'delta_py' in pg:
                                        return 'pixel'
                                    if 'delta_wire' in pg:
                                        return 'wire'
                            elif 'delta_py' in vol:
                                return 'pixel'
                            elif 'delta_wire' in vol:
                                return 'wire'
                        break
            except Exception as e:
                log.warning("readout detection failed on %s: %s", path, e)
                continue
        return 'wire'

    def _plane_has_payload(self, g):
        """True if this group contains a plane's sparse data payload."""
        if self.readout_type == 'pixel':
            return 'delta_py' in g
        return 'delta_wire' in g

    def _iter_planes(self, evt):
        """Yield (plane_label, h5py.Group) for each plane in an event.

        Handles both formats:
          - Old: planes directly under event (east_U, east_V, ...)
          - New: planes under volume_N/ subgroups
        """
        for key in evt:
            obj = evt[key]
            if not isinstance(obj, h5py.Group):
                continue
            if key.startswith('volume_'):
                vol_label = key
                for plane_key in obj:
                    pg = obj[plane_key]
                    if isinstance(pg, h5py.Group) and self._plane_has_payload(pg):
                        yield f'{vol_label}_{plane_key}', pg
            elif self._plane_has_payload(obj):
                yield key, obj

    def _decode_plane_wire(self, g):
        """Decode one wire plane's delta-encoded sparse data."""
        wire_start = int(g.attrs['wire_start'])
        time_start = int(g.attrs['time_start'])

        # cumsum directly into int32 + in-place add avoids the wide-int64
        # accumulator and the extra full-size copies of the old expression.
        wire = np.cumsum(g['delta_wire'][:], dtype=np.int32); wire += wire_start
        time = np.cumsum(g['delta_time'][:], dtype=np.int32); time += time_start

        values = self._decode_values(g)
        return wire, time, values

    def _decode_plane_pixel(self, g):
        """Decode one pixel plane's delta-encoded sparse data."""
        py_start = int(g.attrs['py_start'])
        pz_start = int(g.attrs['pz_start'])
        time_start = int(g.attrs['time_start'])

        py = np.cumsum(g['delta_py'][:], dtype=np.int32); py += py_start
        pz = np.cumsum(g['delta_pz'][:], dtype=np.int32); pz += pz_start
        time = np.cumsum(g['delta_time'][:], dtype=np.int32); time += time_start

        values = self._decode_values(g)
        return py, pz, time, values

    def _decode_values(self, g):
        """Shared value decoding (handles uint16 digitization)."""
        raw_values = g['values'][:]
        vals = raw_values.astype(np.float32)
        if self.decode_digitization and raw_values.dtype == np.uint16:
            vals -= int(g.attrs.get('pedestal', 0))  # in-place, no extra copy
        return vals

    def read_event(self, idx):
        """Read one event, return dict with plane-namespaced sparse arrays.

        Wire returns keys like:
            sensor.{plane}.{wire, time, value}
        Pixel returns keys like:
            sensor.{plane}.{py, pz, time, value}
        """
        if not self._initted:
            self.h5py_worker_init()

        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        data_dict = {}
        for plane_label, pg in self._iter_planes(evt):
            if self.planes != 'all' and plane_label not in self.planes:
                continue

            prefix = f'sensor.{plane_label}'
            if self.readout_type == 'pixel':
                py, pz, time, values = self._decode_plane_pixel(pg)
                data_dict[f'{prefix}.py'] = py
                data_dict[f'{prefix}.pz'] = pz
                data_dict[f'{prefix}.time'] = time
                data_dict[f'{prefix}.value'] = values
            else:
                wire, time, values = self._decode_plane_wire(pg)
                data_dict[f'{prefix}.wire'] = wire
                data_dict[f'{prefix}.time'] = time
                data_dict[f'{prefix}.value'] = values

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

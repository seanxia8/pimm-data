"""
JAXTPCDataset — multimodal dataset for LArTPC detector simulation output.

Loads from co-indexed HDF5 files produced by JAXTPC's production pipeline:

* ``edep/`` — 3D truth deposits
* ``sensor/`` — raw sparse wire / pixel readout
* ``hits/`` — per-instance sensor decomposition
* ``labl/`` — track_id → label lookup tables

Modality strings: ``'edep'``, ``'sensor'``, ``'hits'``, ``'labl'``. See
README §Modality combinations for the combination matrix.

Returns a **nested** dict: each loaded modality owns a sub-dict with clean
unprefixed keys::

    {
      'edep':   {'coord': (N,3), 'energy': (N,1), 'volume_id': ..., ...},
      'sensor': {'coord': (M,D), 'energy': (M,1), 'plane_id': ...,
                 'readout_type': 'wire'|'pixel',
                 'raw': {plane_label: {'wire'|'py'+'pz', 'time', 'value'}}},
      'hits':   {'coord': (E,D), 'energy': (E,1), 'instance': ..., ...,
                 'readout_type': 'wire'|'pixel',
                 'raw': {plane_label: {'wire'|'py'+'pz', 'time',
                                       'group_id', 'charge'}}},
      'labl':   {'v0': {'track_ids': (T,), 'track_pdg': (T,),
                        'deposit_to_track': (N_v,), ...}, 'v1': {...}},
      'bridges':{'group_to_track_v0': (G,), 'deposit_to_group_v0': (N_v,),
                 'qs_fractions_v0': ..., ...},
      'name': str, 'split': str,
    }

Missing modalities have no top-level key. There is no bare ``coord`` / no
precedence / no prefixed aliases — transforms pick a stream explicitly (see
``ApplyToStream`` and ``Collect(stream=...)``).

Registered in :data:`pimm_data.DATASETS` for config-driven construction
via ``dict(type="JAXTPCDataset", ...)``.
"""

import os
import logging
from copy import deepcopy

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
from ._joint_index import build_joint_index
from .readers.jaxtpc_edep import JAXTPCEdepReader
from .readers.jaxtpc_sensor import JAXTPCSensorReader
from .readers.jaxtpc_labl import JAXTPCLablReader
from .readers.jaxtpc_hits import JAXTPCHitsReader

log = logging.getLogger(__name__)


@DATASETS.register_module()
class JAXTPCDataset(DefaultDataset):
    """LArTPC multimodal dataset with nested per-stream output.

    Parameters
    ----------
    data_root : str
        Root directory with ``edep/``, ``sensor/``, ``hits/``, ``labl/``
        subdirectories.
    split : str
        Split name for file discovery.
    modalities : tuple[str]
        Any subset of ``'edep'``, ``'sensor'``, ``'hits'``, ``'labl'``.
        ``('labl',)`` and ``('sensor', 'labl')`` are invalid (see
        README §Modality combinations).
    dataset_name : str
        File prefix (e.g., ``'sim'`` for ``sim_edep_0000.h5``).
    volume : int or None
        Load only this volume index. ``None`` = all volumes.
    label_key : str
        Which labl column to decorate the point clouds with. Must match a
        column in the labl files (``'pdg'``, ``'cluster'``, ``'interaction'``,
        ``'ancestor'``). Raw values from ``track_{label_key}`` are broadcast
        to each deposit / pixel entry; use a downstream ``RemapSegment`` to
        map raw values to task-specific class indices.
    min_deposits : int
        Minimum 3D deposits per event (edep reader filter).
    include_physics : bool
        Whether edep reader loads dx, theta, phi, charge, photons, etc.
    label_keys : list or None
        Which label datasets to load from labl files (None → all).
    transform : list or None
        Transform pipeline.
    test_mode, test_cfg, loop, max_len, ignore_index, cache : standard
        :class:`DefaultDataset` parameters.
    """


    def __init__(
        self,
        data_root,
        split='train',
        modalities=('edep',),
        dataset_name='sim',
        volume=None,
        label_key='pdg',
        label_config=None,
        min_deposits=0,
        include_physics=True,
        label_keys=None,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        cache=False,
        strict_lengths=False,
    ):
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)

        # A3: min_deposits filters on edep deposit counts, so it is a no-op
        # (and silently so) without the edep modality. Fail loud instead.
        if min_deposits > 0 and 'edep' not in self._modalities:
            raise ValueError(
                f"min_deposits={min_deposits} filters on edep deposit counts "
                f"but modalities={self._modalities} does not include 'edep'. "
                "Add 'edep' to modalities or set min_deposits=0.")

        self._dataset_name = dataset_name
        self._volume = volume
        self._label_key = label_key
        self._label_config = label_config
        self._validate_label_config()
        self._min_deposits = min_deposits
        self._include_physics = include_physics
        self._label_keys = label_keys
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        self._source_split = split

        self.edep_reader = None
        self.sensor_reader = None
        self.labl_reader = None
        self.hits_reader = None

        if 'edep' in self._modalities:
            self.edep_reader = JAXTPCEdepReader(
                data_root=self._modality_root('edep'), split=split,
                dataset_name=dataset_name, min_deposits=min_deposits,
                include_physics=include_physics, volume=volume)

        # sensor/hits readout auto-detection. Build the readers unfiltered
        # first so they can detect readout_type, then apply the volume
        # plane filter using the correct plane labels for this readout.
        if 'sensor' in self._modalities:
            self.sensor_reader = JAXTPCSensorReader(
                data_root=self._modality_root('sensor'), split=split,
                dataset_name=dataset_name, planes='all')

        if 'labl' in self._modalities:
            self.labl_reader = JAXTPCLablReader(
                data_root=self._modality_root('labl'), split=split,
                dataset_name=dataset_name, label_keys=label_keys)

        if 'hits' in self._modalities:
            self.hits_reader = JAXTPCHitsReader(
                data_root=self._modality_root('hits'), split=split,
                dataset_name=dataset_name, planes='all')

        # Resolve readout_type once from whichever reader can tell us.
        self._readout_type = 'wire'
        for r in (self.sensor_reader, self.hits_reader):
            if r is not None:
                self._readout_type = r.readout_type
                break

        # Now that readout_type is known, apply per-volume plane filter.
        if volume is not None:
            if self._readout_type == 'pixel':
                planes = [f'volume_{volume}_Pixel']
            else:
                planes = [f'volume_{volume}_U', f'volume_{volume}_V',
                          f'volume_{volume}_Y']
            if self.sensor_reader is not None:
                self.sensor_reader.planes = planes
            if self.hits_reader is not None:
                self.hits_reader.planes = planes

        self._canonical_reader = (self.edep_reader or self.sensor_reader
                                  or self.hits_reader or self.labl_reader)
        # Phase A / D42: build ONE joint cross-modality event index and inject
        # it into every reader, so a single global idx maps to the SAME
        # physics event in all modalities. (Replaces the old
        # `_n_events = min(len(r) ...)`, which left each reader mapping idx
        # through its own present-key index → silent desync under
        # min_deposits>0 or a gap present in some-but-not-all modalities.)
        self._build_joint_index()

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, test_mode=test_mode, test_cfg=test_cfg,
            cache=cache, ignore_index=ignore_index, loop=loop,
        )

        # Fail fast on empty data_list — otherwise get_data() crashes later
        # with an opaque ZeroDivisionError on `idx % len(self.data_list)`.
        if len(self.data_list) == 0:
            raise ValueError(
                f"JAXTPCDataset(data_root={data_root!r}) yielded 0 events "
                f"after filters (min_deposits={min_deposits}, "
                f"max_len={max_len}). Lower min_deposits or verify the "
                f"dataset has events meeting it.")

    @staticmethod
    def _validate_modalities(modalities):
        mods = set(modalities)
        if not mods:
            raise ValueError("modalities is empty; must load at least one")
        unknown = mods - {'edep', 'sensor', 'hits', 'labl'}
        if unknown:
            raise ValueError(
                f"Unknown modalities {unknown}; valid: "
                "'edep', 'sensor', 'hits', 'labl'")
        if mods == {'labl'}:
            raise ValueError(
                "Invalid modality combination ('labl',): labl is a "
                "dimension table and requires an instance-bearing modality "
                "('edep' or 'hits') to join against. "
                "See README §Modality combinations.")
        if mods == {'sensor', 'labl'}:
            raise ValueError(
                "Invalid modality combination ('sensor', 'labl'): sensor has "
                "no instance separation, so labl cannot be attached. Add "
                "'hits' or 'edep' to the modalities tuple. "
                "See README §Modality combinations.")

    def _modality_root(self, modality):
        mod_dir = os.path.join(self._source_data_root, modality)
        if os.path.isdir(mod_dir):
            return mod_dir
        return self._source_data_root

    def _build_joint_index(self):
        """Build one joint cross-modality event index; inject into all readers.

        Delegates to :func:`pimm_data._joint_index.build_joint_index`. See
        that module for the desync this prevents (Phase A / D42).
        """
        named = [(n, r) for n, r in (
            ('edep', self.edep_reader), ('sensor', self.sensor_reader),
            ('hits', self.hits_reader), ('labl', self.labl_reader))
            if r is not None]
        self._n_events = build_joint_index(
            named, strict_lengths=self._strict_lengths,
            source_label=f"JAXTPCDataset({self._source_data_root!r})",
            filter_label=(f"min_deposits={self._min_deposits}"
                          if self._min_deposits > 0 else ''))

    def get_data_list(self):
        n = getattr(self, '_n_events', 0)
        max_len = getattr(self, '_max_len', -1)
        if max_len > 0:
            n = min(n, max_len)
        return list(range(n))

    def get_data(self, idx):
        """Load one event as a nested dict (schema: see module docstring)."""
        real_idx = idx % len(self.data_list)

        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }

        labl_by_volume = {}
        if self.labl_reader is not None:
            labl_by_volume = self._build_labl(self.labl_reader.read_event(real_idx))
            if self._volume is not None:
                # Drop labl volumes the user isn't loading (keeps the
                # dataset's view consistent with ``volume=`` on other readers).
                keep = f'v{self._volume}'
                labl_by_volume = {k: v for k, v in labl_by_volume.items()
                                  if k == keep}
            if labl_by_volume:
                data['labl'] = labl_by_volume

        if self.hits_reader is not None:
            hits_raw = self.hits_reader.read_event(real_idx)
            data['hits'] = self._build_hits_cloud(hits_raw, labl_by_volume)
            bridges = self._build_bridges(hits_raw)
            if bridges:
                data['bridges'] = bridges

        if self.sensor_reader is not None:
            data['sensor'] = self._build_sensor_cloud(
                self.sensor_reader.read_event(real_idx))

        if self.edep_reader is not None:
            data['edep'] = self._build_edep_cloud(
                self.edep_reader.read_event(real_idx), labl_by_volume)

        return data

    # ------------------------------------------------------------------
    # Per-modality builders
    # ------------------------------------------------------------------

    def _build_edep_cloud(self, edep_raw, labl_by_volume):
        """3D deposit sub-dict; decorates with segment/instance if labl present."""
        sub = {}
        for k, v in edep_raw.items():
            sub[k] = v  # coord, energy, volume_id, physics — readers emit bare

        if labl_by_volume and 'volume_id' in sub:
            segment, instance = self._decorate_edep_from_labl(
                sub['volume_id'], labl_by_volume)
            sub['segment'] = segment
            sub['instance'] = instance
            for out, kind, lk in self._track_axes():
                if kind == 'self':
                    sub[out] = instance[:, None]   # the per-deposit track id
                    continue
                seg_axis, _ = self._decorate_edep_from_labl(
                    sub['volume_id'], labl_by_volume, label_key=lk)
                sub[out] = seg_axis[:, None]

        return sub

    def _build_sensor_cloud(self, sensor_raw):
        """Merge per-plane sensor raw into a point cloud + raw passthrough."""
        coord_keys = self._coord_keys()
        planes, coord, energy, plane_id, raw = self._merge_plane_dotted(
            sensor_raw, prefix='sensor', value_key='value',
            coord_keys=coord_keys)
        return {
            'coord': coord, 'energy': energy, 'plane_id': plane_id,
            'planes': planes, 'raw': raw,
            'readout_type': self._readout_type,
        }

    def _build_hits_cloud(self, hits_raw, labl_by_volume):
        """Merge per-plane hits raw into a point cloud + raw passthrough.

        Attaches ``segment`` when labl available (via group_to_track chain).
        ``instance`` is always attached (== group_id).
        """
        coord_keys = self._coord_keys()
        planes, coord, energy, plane_id, raw = self._merge_plane_dotted(
            hits_raw, prefix='hits', value_key='charge',
            coord_keys=coord_keys, extra_keys=('group_id',))
        # instance = per-entry group_id (already int32 from the reader, so
        # no astype copy needed)
        instance = (np.concatenate([raw[p]['group_id'] for p in planes], axis=0)
                    if planes else np.zeros(0, dtype=np.int32))

        sub = {
            'coord': coord, 'energy': energy, 'plane_id': plane_id,
            'instance': instance, 'planes': planes, 'raw': raw,
            'readout_type': self._readout_type,
        }
        if labl_by_volume:
            sub['segment'] = self._decorate_hits_from_labl(
                planes, raw, hits_raw, labl_by_volume)
            for out, kind, lk in self._track_axes():
                if kind == 'self':
                    sub[out] = instance[:, None]   # bare instance (group_id)
                    continue
                sub[out] = self._decorate_hits_from_labl(
                    planes, raw, hits_raw, labl_by_volume,
                    label_key=lk)[:, None]
        return sub

    def _coord_keys(self):
        """Per-plane column names that build the sensor/hits coord vector."""
        if self._readout_type == 'pixel':
            return ('py', 'pz', 'time')
        return ('wire', 'time')

    def _build_labl(self, labl_flat):
        """Convert flat labl_v{N}_col keys into nested {v{N}: {col: arr}}."""
        by_volume = {}
        for k, v in labl_flat.items():
            # Key format: labl_v{idx}_{col}; col may contain underscores
            assert k.startswith('labl_v'), k
            rest = k[len('labl_v'):]
            # Split on first underscore after idx
            idx_end = 0
            while idx_end < len(rest) and rest[idx_end].isdigit():
                idx_end += 1
            vid = 'v' + rest[:idx_end]
            col = rest[idx_end + 1:]  # skip the separator underscore
            by_volume.setdefault(vid, {})[col] = v
        return by_volume

    def _build_bridges(self, hits_raw):
        """Extract per-volume bridge arrays (g2t, deposit_to_group, qs_fractions).

        Under a ``volume=`` filter (F13), keep only that volume's bridges — the
        hits reader still loads every volume's ``*_v{N}`` tables, but the other
        volumes' points are not loaded, so carrying their group machinery is
        wasted payload that confuses downstream consumers."""
        want = None if self._volume is None else str(self._volume)
        bridges = {}
        for k, v in hits_raw.items():
            if (k.startswith('group_to_track_v')
                    or k.startswith('deposit_to_group_v')
                    or k.startswith('qs_fractions_v')):
                if want is not None and k.rsplit('_v', 1)[1] != want:
                    continue
                bridges[k] = v
        return bridges

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_plane_dotted(raw_dict, prefix, value_key, coord_keys,
                            extra_keys=()):
        """Merge `{prefix}.{plane}.{col}` flat keys into a single point cloud.

        ``coord_keys`` is the ordered tuple of per-plane column names that
        form the coord vector (e.g. ``('wire', 'time')`` for wire readout,
        ``('py', 'pz', 'time')`` for pixel). Coord shape is ``(M, len(coord_keys))``.

        Returns (planes, coord, energy, plane_id, raw_nested).
        raw_nested is ``{plane_label: {*coord_keys, value_key, *extra_keys}}``.
        """
        coord_dim = len(coord_keys)
        # Discover planes by scanning for any coord-key suffix (they all
        # share the same plane set, so the first key is enough).
        first_key = coord_keys[0]
        planes = sorted(set(
            k.split('.')[1] for k in raw_dict
            if k.startswith(prefix + '.') and k.endswith('.' + first_key)
        ))
        # Gather per-plane columns + sizes (and build raw_nested) in one pass,
        # then fill preallocated output arrays per-plane. This avoids the
        # np.stack-per-plane + concatenate-across-planes copies (the single
        # biggest reader op on large events); the slice assignment casts
        # int -> float32 in place.
        raw_nested = {}
        plane_cols, sizes = [], []
        for plane in planes:
            cols_arrays = [raw_dict[f'{prefix}.{plane}.{ck}']
                           for ck in coord_keys]
            value = raw_dict[f'{prefix}.{plane}.{value_key}']
            plane_cols.append((cols_arrays, value))
            sizes.append(len(cols_arrays[0]))
            cols = {ck: arr for ck, arr in zip(coord_keys, cols_arrays)}
            cols[value_key] = value
            for ek in extra_keys:
                cols[ek] = raw_dict[f'{prefix}.{plane}.{ek}']
            raw_nested[plane] = cols

        N = sum(sizes)
        coord = np.empty((N, coord_dim), dtype=np.float32)
        energy = np.empty((N, 1), dtype=np.float32)
        plane_id = np.empty((N, 1), dtype=np.int32)
        off = 0
        for i, ((cols_arrays, value), n) in enumerate(zip(plane_cols, sizes)):
            sl = slice(off, off + n)
            for j, arr in enumerate(cols_arrays):
                coord[sl, j] = arr
            energy[sl, 0] = value
            plane_id[sl, 0] = i
            off += n

        return planes, coord, energy, plane_id, raw_nested

    def _validate_label_config(self):
        """Reject ``label_config`` specs the JAXTPC path can't honor (F4).

        JAXTPC labl is per-volume ``track`` tables value-keyed by ``track_ids``
        — there is no event-level or ``particle`` table as in LUCiD. So a spec
        that LUCiD's shared ``decorate_labels`` would satisfy (event scope,
        ``('particle'/'event', col)`` source) has no JAXTPC analog. Raise here
        rather than silently dropping the axis, so the same ``label_config``
        either produces the named key on both detectors or fails loud — never
        diverges quietly. Supported: ``scope='point'`` with ``source='self'``
        (emit the per-point track id) or ``source=('track', col)`` (value-keyed
        gather of ``track_<col>``)."""
        if not self._label_config:
            return
        for spec in self._label_config:
            out = spec.get('out', '?')
            scope = spec.get('scope', 'point')
            if scope != 'point':
                raise ValueError(
                    f"JAXTPCDataset label_config[{out!r}]: scope={scope!r} "
                    "unsupported (no event-level labl table); use scope='point'.")
            src = spec.get('source', 'self')
            ok = (src == 'self'
                  or (isinstance(src, (tuple, list)) and len(src) == 2
                      and src[0] == 'track'))
            if not ok:
                raise ValueError(
                    f"JAXTPCDataset label_config[{out!r}]: source={src!r} "
                    "unsupported; use 'self' or ('track', <column>). "
                    "(LUCiD's ('particle'/'event', …) tables have no JAXTPC "
                    "analog — this would silently drop the axis.)")
            kb = spec.get('keyed_by')
            if kb is not None and kb != 'track_ids':
                raise ValueError(
                    f"JAXTPCDataset label_config[{out!r}]: keyed_by={kb!r} "
                    "unsupported; JAXTPC track columns are keyed by 'track_ids'.")

    def _track_axes(self):
        """``(out_key, kind, column)`` for each point-scope ``label_config`` axis.

        ``kind='self'`` → emit the bare per-modality ``instance`` axis under the
        named key (parity with LUCiD ``source='self'``, where ``self`` == the
        ``instance`` identifier); ``kind='track'`` → value-keyed
        per-volume gather of ``track_<column>`` to emit named schema keys (e.g.
        ``segment_pid`` from ``track_pdg``). Validated by
        :meth:`_validate_label_config`, so unsupported specs raise at
        construction and are never silently dropped here."""
        if not self._label_config:
            return []
        axes = []
        for spec in self._label_config:
            if spec.get('scope', 'point') != 'point':
                continue
            src = spec.get('source', 'self')
            if src == 'self':
                axes.append((spec['out'], 'self', None))
            elif (isinstance(src, (tuple, list)) and len(src) == 2
                    and src[0] == 'track'):
                axes.append((spec['out'], 'track', src[1]))
        return axes

    def _decorate_edep_from_labl(self, volume_id, labl_by_volume,
                                 label_key=None):
        """Broadcast per-track labl data onto each edep deposit.

        Uses ``labl[vN]['deposit_to_track']`` (row-aligned to the volume's
        edep deposits) as the per-deposit FK, then looks up
        ``labl[vN]['track_{label_key}']`` via binary search on ``track_ids``.
        ``label_key`` defaults to ``self._label_key``; pass an explicit key to
        gather a different axis (used by ``label_config``).
        """
        vid_flat = volume_id.ravel()
        n_total = vid_flat.shape[0]
        instance = np.full(n_total, -1, dtype=np.int32)
        segment = np.full(n_total, -1, dtype=np.int32)
        meta_col = f'track_{label_key or self._label_key}'

        for vkey, vdata in labl_by_volume.items():
            vol_num = int(vkey[1:])
            mask = vid_flat == vol_num
            if not mask.any():
                continue
            if 'deposit_to_track' not in vdata:
                continue
            per_dep_tid = vdata['deposit_to_track'].astype(np.int32)
            n_vol = int(mask.sum())
            if per_dep_tid.shape[0] != n_vol:
                log.warning("labl.%s.deposit_to_track len %d != edep vol %d len %d",
                            vkey, per_dep_tid.shape[0], vol_num, n_vol)
                continue
            instance[mask] = per_dep_tid

            if 'track_ids' in vdata and meta_col in vdata:
                tids = vdata['track_ids']
                vals = vdata[meta_col]
                order = np.argsort(tids)
                s_tids = tids[order]
                s_vals = vals[order]
                pos = np.searchsorted(s_tids, per_dep_tid)
                pos = np.clip(pos, 0, len(s_tids) - 1)
                matched = s_tids[pos] == per_dep_tid
                segment[mask] = np.where(matched, s_vals[pos], -1)

        return segment, instance

    def _decorate_hits_from_labl(self, planes, raw_nested, hits_flat,
                                 labl_by_volume, label_key=None):
        """Per-hits-entry segment label via group_to_track → track lookup.

        ``label_key`` defaults to ``self._label_key``; pass an explicit key to
        gather a different axis (used by ``label_config``)."""
        meta_col = f'track_{label_key or self._label_key}'
        all_labels = []
        for plane in planes:
            cols = raw_nested[plane]
            gid = cols['group_id']
            # plane label is 'volume_{v}_{U|V|Y}' — extract volume index
            vol_idx_str = plane.split('_')[1]
            vkey = f'v{vol_idx_str}'

            n = gid.shape[0]
            labels = np.full(n, -1, dtype=np.int32)

            g2t_key = f'group_to_track_v{vol_idx_str}'
            g2t = hits_flat.get(g2t_key)
            if g2t is None or vkey not in labl_by_volume:
                all_labels.append(labels)
                continue
            vdata = labl_by_volume[vkey]
            if 'track_ids' not in vdata or meta_col not in vdata:
                all_labels.append(labels)
                continue

            valid = (gid >= 0) & (gid < len(g2t))
            tids = np.where(valid, g2t[gid], -1)
            labl_tids = vdata['track_ids']
            labl_vals = vdata[meta_col]
            order = np.argsort(labl_tids)
            s_tids = labl_tids[order]
            s_vals = labl_vals[order]
            pos = np.searchsorted(s_tids, tids)
            pos = np.clip(pos, 0, len(s_tids) - 1)
            matched = s_tids[pos] == tids
            labels[matched] = s_vals[pos[matched]]
            all_labels.append(labels)

        if not all_labels:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate(all_labels, axis=0)

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx = int(np.searchsorted(reader.cumulative_lengths, idx, side='right'))
        local = idx - (int(reader.cumulative_lengths[file_idx - 1])
                       if file_idx > 0 else 0)
        event_num = reader.indices[file_idx][local]
        fname = os.path.basename(reader.h5_files[file_idx])
        return f"{fname}_evt{event_num:03d}"

    def prepare_test_data(self, idx):
        """Test-time data prep.

        Expects ``segment`` to be produced at the top level by a terminal
        :class:`Collect` transform (e.g. ``Collect(stream='edep', ...)``).
        """
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = dict(name=data_dict.pop("name"))
        if "segment" in data_dict:
            result_dict["segment"] = data_dict.pop("segment")
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        data_dict_list = [aug(deepcopy(data_dict)) for aug in self.aug_transform]
        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part
        fragment_list = [self.post_transform(f) for f in fragment_list]
        result_dict["fragment_list"] = fragment_list
        return result_dict

    def __del__(self):
        for attr in ('edep_reader', 'sensor_reader', 'labl_reader',
                     'hits_reader'):
            reader = getattr(self, attr, None)
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

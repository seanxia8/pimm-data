"""
LUCiDDataset — multimodal dataset for LUCiD Water Cherenkov simulation
output (``format_version: 3``).

Loads from co-indexed per-modality HDF5 shards:

* ``edep/``   — 3D Geant4 step deposits
* ``sensor/`` — sparse PMT response (event-level aggregate of ``hits``)
* ``hits/``   — per-particle PMT hit decomposition
* ``labl/``   — per-event / per-particle / per-track label tables

Returns a **nested** dict: each loaded modality owns a sub-dict with
clean, unprefixed keys::

    {
      'edep':   {'coord': (N,3), 'energy': (N,1), 'time': (N,1),
                 'track_idx': (N,), 'direction': ..., 'beta_start': ...,
                 'n_cherenkov': ..., 'instance': (N,), 'segment': (N,)},
      'sensor': {'coord': (H,3), 'energy': (H,1), 'time': (H,1),
                 'sensor_idx': (H,)},
      'hits':   {'coord': (E,3), 'energy': (E,1), 'time': (E,1),
                 'sensor_idx': (E,), 'particle_idx': (E,),
                 'instance': (E,), 'segment': (E,)},
      'labl':   {'event': {...}, 'particle': {...}, 'track': {...}},
      'name': str, 'split': str,
    }

Instance / segment labels (``hits`` and ``edep``) are *particle-level* by
default: ``instance = particle_idx`` and ``segment = per_particle.category``.
For coarser groupings (ancestor-level), use
``labl.particle.ancestor_particle_idx`` (or ``labl.track.ancestor_particle_idx``
for edep) in a downstream transform — this is a one-line lookup, so we keep
the dataset free of grouping policy.

Missing modalities have no top-level key. Two modality combinations are
rejected: ``('labl',)`` alone, and ``('sensor', 'labl')``. ``labl`` is a
dimension table and needs an instance-bearing modality (``edep`` or
``hits``) to attach to.

Registered in :data:`pimm_data.DATASETS`.
"""

import os
import logging
from copy import deepcopy

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset
from ._joint_index import build_joint_index
from ._label_decorate import decorate_labels
from .readers.lucid_edep import LUCiDEdepReader
from .readers.lucid_sensor import LUCiDSensorReader
from .readers.lucid_hits import LUCiDHitsReader
from .readers.lucid_labl import LUCiDLablReader

log = logging.getLogger(__name__)

_VALID_MODALITIES = {'edep', 'sensor', 'hits', 'labl'}


@DATASETS.register_module()
class LUCiDDataset(DefaultDataset):
    """Water Cherenkov multimodal dataset with nested per-stream output.

    Parameters
    ----------
    data_root : str
        Root directory containing per-modality subdirectories.
    split : str
        Split name for file discovery.
    modalities : tuple[str]
        Any subset of ``{'edep', 'sensor', 'hits', 'labl'}``.
        ``('labl',)`` and ``('sensor', 'labl')`` are invalid.
    dataset_name : str
        File prefix (e.g. ``'wc'`` matches ``wc_edep_0000.h5``).
    min_segments : int
        Drop events with fewer than this many edep segments (edep only).
    include_physics : bool
        Whether edep emits direction / beta_start / n_cherenkov.
    pe_threshold : float
        Drop hits entries with ``pe <= pe_threshold`` (hits only).
    pmt_positions, pmt_positions_file : optional
        Overrides for sensor geometry — normally the file's
        ``config/sensor_positions`` is used.
    transform, test_mode, test_cfg, loop, max_len, ignore_index, cache :
        Standard :class:`DefaultDataset` parameters.
    """

    def __init__(
        self,
        data_root,
        split='',
        modalities=('sensor',),
        dataset_name='wc',
        min_segments=0,
        include_physics=True,
        pe_threshold=0.0,
        pmt_positions=None,
        pmt_positions_file=None,
        label_config=None,
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

        # A3: min_segments filters on edep segment counts, so it is a silent
        # no-op without the edep modality. Fail loud instead.
        if min_segments > 0 and 'edep' not in self._modalities:
            raise ValueError(
                f"min_segments={min_segments} filters on edep segment counts "
                f"but modalities={self._modalities} does not include 'edep'. "
                "Add 'edep' to modalities or set min_segments=0.")

        self._dataset_name = dataset_name
        self._min_segments = min_segments
        self._label_config = label_config
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        self._source_split = split

        self.edep_reader = None
        self.sensor_reader = None
        self.hits_reader = None
        self.labl_reader = None

        if 'edep' in self._modalities:
            self.edep_reader = LUCiDEdepReader(
                data_root=self._modality_root('edep'), split=split,
                dataset_name=dataset_name, min_segments=min_segments,
                include_physics=include_physics)

        if 'sensor' in self._modalities:
            self.sensor_reader = LUCiDSensorReader(
                data_root=self._modality_root('sensor'), split=split,
                dataset_name=dataset_name,
                pmt_positions=pmt_positions,
                pmt_positions_file=pmt_positions_file)

        if 'hits' in self._modalities:
            self.hits_reader = LUCiDHitsReader(
                data_root=self._modality_root('hits'), split=split,
                dataset_name=dataset_name, pe_threshold=pe_threshold)

        if 'labl' in self._modalities:
            self.labl_reader = LUCiDLablReader(
                data_root=self._modality_root('labl'), split=split,
                dataset_name=dataset_name)

        self._canonical_reader = (self.edep_reader or self.hits_reader
                                  or self.sensor_reader or self.labl_reader)
        # Phase A / D42: one joint cross-modality event index injected into
        # every reader, so a global idx maps to the SAME physics event in all
        # modalities (replaces `_n_events = min(len(r) ...)`, which desynced
        # under min_segments>0 or a gap present in some-but-not-all modalities).
        named = [(n, r) for n, r in (
            ('edep', self.edep_reader), ('sensor', self.sensor_reader),
            ('hits', self.hits_reader), ('labl', self.labl_reader))
            if r is not None]
        self._n_events = build_joint_index(
            named, strict_lengths=strict_lengths,
            source_label=f"LUCiDDataset({data_root!r})",
            filter_label=(f"min_segments={min_segments}"
                          if min_segments > 0 else ''))

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, test_mode=test_mode, test_cfg=test_cfg,
            cache=cache, ignore_index=ignore_index, loop=loop,
        )

    @staticmethod
    def _validate_modalities(modalities):
        mods = set(modalities)
        if not mods:
            raise ValueError("modalities is empty; must load at least one")
        unknown = mods - _VALID_MODALITIES
        if unknown:
            raise ValueError(
                f"Unknown modalities {unknown}; valid: {_VALID_MODALITIES}")
        if mods == {'labl'}:
            raise ValueError(
                "Invalid modality combination ('labl',): labl is a "
                "dimension table and requires 'edep' or 'hits' to attach to.")
        if mods == {'sensor', 'labl'}:
            raise ValueError(
                "Invalid modality combination ('sensor', 'labl'): sensor "
                "has no particle separation — labl can't be attached. Add "
                "'hits' or 'edep' to the modalities tuple.")

    def _modality_root(self, modality):
        mod_dir = os.path.join(self._source_data_root, modality)
        if os.path.isdir(mod_dir):
            return mod_dir
        return self._source_data_root

    def get_data_list(self):
        n = getattr(self, '_n_events', 0)
        max_len = getattr(self, '_max_len', -1)
        if max_len > 0:
            n = min(n, max_len)
        return list(range(n))

    def get_data(self, idx):
        real_idx = idx % len(self.data_list)

        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }

        labl = None
        if self.labl_reader is not None:
            labl = self._build_labl(self.labl_reader.read_event(real_idx))
            data['labl'] = labl

        if self.sensor_reader is not None:
            data['sensor'] = self._build_sensor(
                self.sensor_reader.read_event(real_idx))

        if self.hits_reader is not None:
            data['hits'] = self._build_hits(
                self.hits_reader.read_event(real_idx), labl)

        if self.edep_reader is not None:
            data['edep'] = self._build_edep(
                self.edep_reader.read_event(real_idx), labl)

        return data

    # ------------------------------------------------------------------
    # Per-modality builders
    # ------------------------------------------------------------------

    def _build_sensor(self, raw):
        """Sparse PMT point cloud: coord indexed by sensor_idx."""
        sensor_idx = raw['sensor_idx']
        pmt_coord = raw.get('pmt_coord')
        if pmt_coord is not None:
            coord = pmt_coord[sensor_idx].astype(np.float32)
        else:
            coord = sensor_idx.astype(np.float32)[:, None]
        return {
            'coord': coord,
            'energy': raw['pmt_pe'][:, None].astype(np.float32),
            'time': raw['pmt_t'][:, None].astype(np.float32),
            'sensor_idx': sensor_idx,
        }

    def _build_hits(self, raw, labl):
        """Per-particle PMT hit point cloud.

        Duplicate points are intentional: same PMT contributed by N
        particles → N rows. ``instance = particle_idx`` tags each row.
        """
        sensor_idx_arr = raw['sensor_idx']
        particle_idx_arr = raw['particle_idx']

        pmt_coord = None
        if self.sensor_reader is not None:
            # Reuse the geometry already loaded by the sensor reader.
            pmt_coord = getattr(self.sensor_reader, '_pmt_positions', None)
        if pmt_coord is None:
            # Pull directly from the hits file's own config group.
            pmt_coord = self._hits_sensor_positions()

        if pmt_coord is not None:
            coord = pmt_coord[sensor_idx_arr].astype(np.float32)
        else:
            coord = sensor_idx_arr.astype(np.float32)[:, None]

        sub = {
            'coord': coord,
            'energy': raw['pe'][:, None].astype(np.float32),
            'time': raw['t'][:, None].astype(np.float32),
            'sensor_idx': sensor_idx_arr,
            'particle_idx': particle_idx_arr,
            'instance': particle_idx_arr.astype(np.int32),
        }
        if labl is not None:
            category = labl['particle'].get('category')
            if category is not None:
                sub['segment'] = self._lookup_per_particle(
                    particle_idx_arr, category)
            if self._label_config is not None:
                decorate_labels(
                    sub, labl,
                    lambda name: (particle_idx_arr
                                  if name == 'particle_idx' else None),
                    self._label_config)
        return sub

    def _build_edep(self, raw, labl):
        """3D deposit cloud decorated with particle-level labels from labl."""
        sub = dict(raw)  # shallow copy; readers emit fresh arrays
        track_idx = sub['track_idx']

        if labl is not None:
            track_tbl = labl['track']
            track_particle_idx = track_tbl.get('particle_idx')
            if track_particle_idx is not None:
                particle_idx = self._lookup_per_track(
                    track_idx, track_particle_idx)
                sub['particle_idx'] = particle_idx
                sub['instance'] = particle_idx
                category = labl['particle'].get('category')
                if category is not None:
                    sub['segment'] = self._lookup_per_particle(
                        particle_idx, category)
            if self._label_config is not None:
                pidx = sub.get('particle_idx')
                decorate_labels(
                    sub, labl,
                    lambda name: (pidx if name == 'particle_idx' else None),
                    self._label_config)
        return sub

    def _build_labl(self, flat):
        """Rebuild nested ``{event, interaction, particle, track}`` dict from
        flat keys (the tables ``decorate_labels`` resolves ``source`` against).
        """
        out = {'event': {}, 'interaction': {}, 'particle': {}, 'track': {}}
        for k, v in flat.items():
            if k.startswith('labl_event_'):
                out['event'][k[len('labl_event_'):]] = v
            elif k.startswith('labl_interaction_'):
                out['interaction'][k[len('labl_interaction_'):]] = v
            elif k.startswith('labl_particle_'):
                out['particle'][k[len('labl_particle_'):]] = v
            elif k.startswith('labl_track_'):
                out['track'][k[len('labl_track_'):]] = v
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _hits_sensor_positions(self):
        """Best-effort fetch of hits's config/sensor_positions.

        The hits file duplicates the sensor geometry; fall back to it
        when no sensor reader is active.
        """
        reader = self.hits_reader
        if reader is None:
            return None
        if not reader._initted:
            reader.h5py_worker_init()
        try:
            cfg = reader._h5data[0]['config']
            if 'sensor_positions' in cfg:
                return cfg['sensor_positions'][:].astype(np.float32)
        except Exception:
            pass
        return None

    @staticmethod
    def _lookup_per_particle(particle_idx, per_particle_col,
                             fill=-1):
        """Gather per-particle values for each row's particle_idx."""
        n = per_particle_col.shape[0]
        valid = (particle_idx >= 0) & (particle_idx < n)
        out = np.full(particle_idx.shape, fill,
                      dtype=per_particle_col.dtype)
        if valid.any():
            out[valid] = per_particle_col[particle_idx[valid]]
        return out

    @staticmethod
    def _lookup_per_track(track_idx, per_track_col, fill=-1):
        """Gather per-track values for each row's track_idx."""
        n = per_track_col.shape[0]
        valid = (track_idx >= 0) & (track_idx < n)
        out = np.full(track_idx.shape, fill, dtype=per_track_col.dtype)
        if valid.any():
            out[valid] = per_track_col[track_idx[valid]]
        return out

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx = int(np.searchsorted(reader.cumulative_lengths, idx,
                                       side='right'))
        local = idx - (int(reader.cumulative_lengths[file_idx - 1])
                       if file_idx > 0 else 0)
        event_num = reader.indices[file_idx][local]
        fname = os.path.basename(reader.h5_files[file_idx])
        return f"{fname}_evt{event_num:03d}"

    def prepare_test_data(self, idx):
        """Test-time data prep.

        Expects a terminal ``Collect`` transform to lift a ``segment``
        stream to the top level (same contract as JAXTPCDataset).
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
        for attr in ('edep_reader', 'sensor_reader',
                     'hits_reader', 'labl_reader'):
            reader = getattr(self, attr, None)
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

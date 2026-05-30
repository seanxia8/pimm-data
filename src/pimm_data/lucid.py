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

import numpy as np

from .builder import DATASETS
from ._dataset_base import _MultiModalShardDataset
from ._label_decorate import decorate_labels, gather_with_fill
from .readers.lucid_edep import LUCiDEdepReader
from .readers.lucid_sensor import LUCiDSensorReader
from .readers.lucid_hits import LUCiDHitsReader
from .readers.lucid_labl import LUCiDLablReader


@DATASETS.register_module()
class LUCiDDataset(_MultiModalShardDataset):
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
        self._build_joint_index(
            source_label=f"LUCiDDataset({data_root!r})",
            filter_label=(f"min_segments={min_segments}"
                          if min_segments > 0 else ''))

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, test_mode=test_mode, test_cfg=test_cfg,
            cache=cache, ignore_index=ignore_index, loop=loop,
        )

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
                sub['segment'] = gather_with_fill(particle_idx_arr, category)
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
                particle_idx = gather_with_fill(track_idx, track_particle_idx)
                sub['particle_idx'] = particle_idx
                sub['instance'] = particle_idx
                category = labl['particle'].get('category')
                if category is not None:
                    sub['segment'] = gather_with_fill(particle_idx, category)
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
        # First shard that actually opened — shard 0 may be a skipped dangling
        # shard (F17); PMT geometry is shared across shards.
        f0 = next((h for h in reader._h5data if h is not None), None)
        if f0 is not None and 'config' in f0 \
                and 'sensor_positions' in f0['config']:
            return f0['config']['sensor_positions'][:].astype(np.float32)
        return None

"""
LUCiDLablReader — per-event labels for LUCiD ``labl/`` HDF5 files
(``format_version: 3``).

The labl file carries three label scopes:

* ``per_event`` — scalar(s) per event (``t0``, ``overall_containment``)
* ``per_particle`` — per-particle-index tables (``category``, ``containment``,
  genealogy CSR)
* ``per_track`` — per-Geant4-track tables (``track_id``, ``pdg``,
  ``parent_id``, ``ancestor`` [root ancestor ``track_id``],
  ``particle_idx`` [FK into ``per_particle``], ``interaction``,
  ``initial_energy``, ``n_cherenkov``)

Derived columns added by the reader (pure reductions, no training
semantics):

* ``labl_track_ancestor_particle_idx`` — ``per_track.ancestor`` resolved
  to a particle_idx via the ``per_track`` table.
* ``labl_particle_ancestor_particle_idx`` — same, broadcast from any
  track of each particle (all tracks in a particle share one root).

These let a downstream transform flip ``instance`` from particle-level
to ancestor-level with a single lookup:

    data['hits']['instance'] = \
        labl['particle']['ancestor_particle_idx'][data['hits']['particle_idx']]

Output dict (flat; dataset layer rebuilds nested ``{event, particle, track}``):

    labl_event_t0                         ()
    labl_event_overall_containment        ()

    labl_particle_category                (P,) int32
    labl_particle_containment             (P,) float32
    labl_particle_genealogy_data          (G,) int32
    labl_particle_genealogy_offsets       (P+1,) int32
    labl_particle_ext_genealogy_data      (Ge,) int32
    labl_particle_ext_genealogy_offsets   (P+1,) int32
    labl_particle_ancestor_particle_idx   (P,) int32   (derived)

    labl_track_track_id                   (T,) int32
    labl_track_pdg                        (T,) int32
    labl_track_parent_id                  (T,) int32
    labl_track_particle_idx               (T,) int32
    labl_track_ancestor                   (T,) int32
    labl_track_interaction                (T,) int32
    labl_track_initial_energy             (T,) float32
    labl_track_n_cherenkov                (T,) int32
    labl_track_ancestor_particle_idx      (T,) int32   (derived)
"""

import os
import glob
import logging
import numpy as np
import h5py

log = logging.getLogger(__name__)


_PARTICLE_KEYS = (
    'category', 'containment',
    'genealogy_data', 'genealogy_offsets',
    'ext_genealogy_data', 'ext_genealogy_offsets',
)
_TRACK_KEYS = (
    'track_id', 'pdg', 'parent_id', 'particle_idx', 'ancestor',
    'interaction', 'initial_energy', 'n_cherenkov',
)
_EVENT_KEYS = ('t0', 'overall_containment')

_INT_KEYS = {'category', 'genealogy_data', 'genealogy_offsets',
             'ext_genealogy_data', 'ext_genealogy_offsets',
             'track_id', 'pdg', 'parent_id', 'particle_idx',
             'ancestor', 'interaction', 'n_cherenkov'}


class LUCiDLablReader:
    """Reads per-event label tables from LUCiD ``labl/`` files.

    Parameters
    ----------
    data_root : str
        Directory containing labl shard files.
    split : str
        Split name (used as subdirectory when present).
    dataset_name : str
        File prefix — matches ``{dataset_name}_labl_*.h5``.
    """

    def __init__(self, data_root, split='', dataset_name='wc', **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name

        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No LUCiD labl files found for '{dataset_name}' in "
            f"{data_root}/{split}")

        self._initted = False
        self._h5data = []
        self._build_index()

    def _find_files(self):
        for pattern in (
            os.path.join(self.data_root, self.split,
                         f'{self.dataset_name}_labl_*.h5'),
            os.path.join(self.data_root, f'{self.dataset_name}_labl_*.h5'),
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
                with h5py.File(h5_path, 'r', libver='latest', swmr=True) as f:
                    n_events = int(f['config'].attrs['n_events'])
                    index = np.arange(n_events, dtype=np.int64)
            except Exception as e:
                log.warning("Error processing %s: %s", h5_path, e)
                index = np.array([], dtype=np.int64)

            self.cumulative_lengths.append(len(index))
            self.indices.append(index)

        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info("LUCiDLablReader: %d events from %d files",
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
        return self._h5data[file_idx], f'event_{event_num:03d}'

    @staticmethod
    def _cast(arr, key):
        if key == 'initial_energy' or key == 'containment':
            return arr.astype(np.float32)
        if key in _INT_KEYS:
            return arr.astype(np.int32)
        return arr.astype(np.float32)

    @staticmethod
    def _derive_ancestor_particle_idx(track_id, particle_idx, ancestor):
        """Map each track's root-ancestor track_id to its particle_idx.

        Returns
        -------
        track_ancestor_pidx : (T,) int32
            Per-track ancestor particle_idx; -1 when the ancestor
            track_id isn't present in ``per_track`` (shouldn't happen
            in well-formed files).
        """
        if track_id.size == 0:
            return np.zeros(0, dtype=np.int32)

        order = np.argsort(track_id, kind='stable')
        sorted_tids = track_id[order]
        sorted_pidx = particle_idx[order]

        pos = np.searchsorted(sorted_tids, ancestor)
        pos = np.clip(pos, 0, sorted_tids.size - 1)
        matched = sorted_tids[pos] == ancestor
        result = np.where(matched, sorted_pidx[pos], -1).astype(np.int32)
        return result

    @staticmethod
    def _collapse_to_particle(track_ancestor_pidx, particle_idx, n_particles):
        """Reduce per-track ancestor particle_idx to per-particle.

        All tracks of a given particle share one root ancestor, so any
        track of the particle gives the same answer. Take the first.
        """
        out = np.full(n_particles, -1, dtype=np.int32)
        if particle_idx.size == 0:
            return out
        uniq, first_idx = np.unique(particle_idx, return_index=True)
        # Mask out any out-of-range particle_idx (defensive).
        valid = (uniq >= 0) & (uniq < n_particles)
        out[uniq[valid]] = track_ancestor_pidx[first_idx[valid]]
        return out

    def read_event(self, idx):
        if not self._initted:
            self.h5py_worker_init()

        f, event_key = self._locate_event(idx)
        evt = f[event_key]

        data = {}

        if 'per_event' in evt:
            ev = evt['per_event']
            for k in _EVENT_KEYS:
                if k in ev:
                    arr = np.asarray(ev[k][()]).astype(np.float32)
                    data[f'labl_event_{k}'] = arr

        pp = evt['per_particle'] if 'per_particle' in evt else None
        n_particles = int(evt.attrs.get('n_particles', 0))
        if pp is not None:
            for k in _PARTICLE_KEYS:
                if k in pp:
                    data[f'labl_particle_{k}'] = self._cast(pp[k][:], k)

        pt = evt['per_track'] if 'per_track' in evt else None
        if pt is not None:
            for k in _TRACK_KEYS:
                if k in pt:
                    data[f'labl_track_{k}'] = self._cast(pt[k][:], k)

            tid = data.get('labl_track_track_id')
            pidx = data.get('labl_track_particle_idx')
            anc = data.get('labl_track_ancestor')
            if tid is not None and pidx is not None and anc is not None:
                track_anc_pidx = self._derive_ancestor_particle_idx(
                    tid, pidx, anc)
                data['labl_track_ancestor_particle_idx'] = track_anc_pidx
                data['labl_particle_ancestor_particle_idx'] = \
                    self._collapse_to_particle(track_anc_pidx, pidx,
                                               n_particles)

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

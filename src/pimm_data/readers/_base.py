"""Shared lifecycle for the sharded HDF5 readers (consolidation).

All eight readers (jaxtpc / lucid × step / hits / sensor / labl) index their
shard files identically: glob by modality, build a per-shard present-event
index (gap-tolerant via :func:`read_shard_meta` — F6), map a global ``idx``
through the cumulative lengths, and lazily open handles (skipping empty /
dangling shards — F17). This base owns that contract so a future index/open
change is one edit, not eight (the structural condition that made F6 and F17
eight-way fixes).

Subclasses set :attr:`_MODALITY` and implement ``read_event``; they override
only the genuine seams: :meth:`_index_for_shard` (the step readers filter by
deposit / segment count) and, for the sensor reader, an extra
:meth:`h5py_worker_init` step to capture PMT geometry.
"""

import glob
import logging
import os

import numpy as np

from .._shard_meta import read_shard_meta, open_event_files

log = logging.getLogger(__name__)

# All shard writers name event groups ``event_NNN`` (≥3 digits; wider for
# event numbers ≥ 1000 — ``:03d`` is a minimum width, not a cap).
EVENT_KEY_FMT = "event_{:03d}"


class ShardReaderBase:
    """Index/locate/open/close lifecycle shared by every shard reader."""

    _MODALITY = None  # 'step' | 'sensor' | 'hits' | 'labl'

    # -- construction -------------------------------------------------------

    def _init_shards(self):
        """Discover shards and build the index.

        Call from ``__init__`` once the subclass has set ``data_root`` /
        ``split`` / ``dataset_name`` (and any of its own attrs the index build
        depends on, e.g. ``min_deposits``).
        """
        self.h5_files = self._find_files()
        assert len(self.h5_files) > 0, (
            f"No {self._MODALITY} files found for '{self.dataset_name}' in "
            f"{self.data_root}/{self.split}")
        self._initted = False
        self._pid = None
        self._h5data = []
        self._build_index()

    def _find_files(self):
        """Glob shards: ``{root}/{split}/{name}_{modality}_*.h5`` then the
        flat ``{root}/{name}_{modality}_*.h5`` fallback."""
        for pattern in (
            os.path.join(self.data_root, self.split,
                         f"{self.dataset_name}_{self._MODALITY}_*.h5"),
            os.path.join(self.data_root,
                         f"{self.dataset_name}_{self._MODALITY}_*.h5"),
        ):
            files = sorted(glob.glob(pattern))
            if files:
                return files
        return []

    # -- index --------------------------------------------------------------

    def _index_for_shard(self, h5_path):
        """Present event numbers for one shard (gap-tolerant — F6).

        Override to filter (step min_deposits / min_segments)."""
        return read_shard_meta(h5_path)["present_events"]

    def _build_index(self):
        self.cumulative_lengths = []
        self.indices = []
        for h5_path in self.h5_files:
            try:
                index = self._index_for_shard(h5_path)
            except Exception as e:
                log.warning("Error processing %s: %s", h5_path, e)
                index = np.array([], dtype=np.int64)
            self.cumulative_lengths.append(len(index))
            self.indices.append(index)
        self.cumulative_lengths = np.cumsum(self.cumulative_lengths)
        log.info("%s: %d events from %d files", type(self).__name__,
                 int(self.cumulative_lengths[-1])
                 if len(self.cumulative_lengths) else 0, len(self.h5_files))

    # -- locate / open ------------------------------------------------------

    def locate(self, idx):
        """Global ``idx`` → ``(file_idx, event_num)``."""
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx,
                                       side="right"))
        base = (int(self.cumulative_lengths[file_idx - 1])
                if file_idx > 0 else 0)
        event_num = int(self.indices[file_idx][idx - base])
        return file_idx, event_num

    def _locate_event(self, idx):
        """Global ``idx`` → ``(file_handle, event_key)``."""
        self._ensure_open()
        file_idx, event_num = self.locate(idx)
        return self._h5data[file_idx], EVENT_KEY_FMT.format(event_num)

    def _ensure_open(self):
        """Open handles in THIS process; reopen after a fork.

        HDF5 file descriptors must not be shared across a fork — if a handle
        was opened in the parent (e.g. a construction-time count scan) the
        DataLoader workers inherit corrupt state. We tag the open with the pid
        and reopen fresh whenever the pid changes (dropping, not closing, the
        inherited handles — the fds belong to the parent)."""
        if self._initted and self._pid == os.getpid():
            return
        if self._initted:                       # stale handles from a parent fork
            self._h5data = []
            self._initted = False
        self.h5py_worker_init()

    def h5py_worker_init(self):
        """Open one handle per shard (None for empty/dangling — F17)."""
        self._h5data = open_event_files(self.h5_files, self.indices)
        self._initted = True
        self._pid = os.getpid()

    # -- size / teardown ----------------------------------------------------

    def __len__(self):
        return (int(self.cumulative_lengths[-1])
                if len(self.cumulative_lengths) > 0 else 0)

    def close(self):
        if self._initted:
            for f in self._h5data:
                try:
                    f.close()
                except Exception:
                    pass
            self._h5data = []
            self._initted = False

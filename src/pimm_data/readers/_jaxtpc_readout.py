"""Shared base for the JAXTPC readout readers (``sensor`` + ``hits``).

Both auto-detect wire vs pixel readout the same way — a ``readout_type``
config-attr fast path (memoized via :func:`read_shard_meta`, no extra open),
with a per-reader plane-dataset scan fallback for files written before the
attr existed — and namespace their per-plane output by plane label.

This base owns the detection *contract* (one definition of how readout_type is
resolved). Each subclass supplies only the fallback ``_scan_readout_type``
(the dataset names it probes differ) and its own per-plane decoders.
"""

import logging

from .._shard_meta import read_shard_meta
from ._base import ShardReaderBase

log = logging.getLogger(__name__)


class JAXTPCReadoutReader(ShardReaderBase):
    """ShardReaderBase + shared wire/pixel readout-type detection."""

    def _detect_readout_type(self):
        """Return ``'wire'`` or ``'pixel'``.

        Fast path: the ``readout_type`` config attr from the cached shard meta
        (F15 — no second file open in the common case). On miss, fall back to
        ``self._scan_readout_type(path)``, which opens the shard and inspects
        plane datasets. Defaults to ``'wire'`` if nothing resolves.
        """
        for path in self.h5_files:
            try:
                rt = str(read_shard_meta(path)['config_attrs'].get(
                    'readout_type', ''))
                if rt in ('wire', 'pixel'):
                    return rt
                rt = self._scan_readout_type(path)
                if rt in ('wire', 'pixel'):
                    return rt
            except Exception as e:
                log.warning("readout detection failed on %s: %s", path, e)
                continue
        return 'wire'

    def _scan_readout_type(self, path):
        """Inspect one shard's plane datasets to infer readout type.

        Subclass-specific: ``sensor`` and ``hits`` store different per-plane
        dataset names. Return ``'wire'``, ``'pixel'``, or ``None``.
        """
        raise NotImplementedError

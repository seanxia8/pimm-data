"""Shared base for the multimodal shard datasets (consolidation).

:class:`~pimm_data.jaxtpc.JAXTPCDataset` and
:class:`~pimm_data.lucid.LUCiDDataset` both compose per-modality shard readers
(edep / sensor / hits / labl), build one joint cross-modality event index
(Phase A / D42), and expose the same minimal ``torch.utils.data.Dataset``
surface (``__getitem__`` → transform pipeline, ``__len__``). That wiring lives
here once; each subclass supplies only its reader construction (``__init__``)
and its per-modality ``_build_*`` cloud builders / label model.

This base is self-contained (it inherits ``torch.utils.data.Dataset``
directly): the detector datasets never used a generic npy loader, cache, or
test-time augmentation path.

A subclass must, in ``__init__`` (before ``super().__init__``): set
``_modalities`` / ``_max_len`` / ``_strict_lengths`` / ``_source_data_root``,
create the ``{edep,sensor,hits,labl}_reader`` attrs (None when absent), set
``_canonical_reader`` (the reader whose file names label events), and call
``self._build_joint_index(source_label, filter_label)``. It must also define
``get_data(idx)`` returning the nested per-modality sample dict.
"""

import os
import logging

from torch.utils.data import Dataset

from .transform import Compose
from ._joint_index import build_joint_index

log = logging.getLogger(__name__)


class ShardEventDataset(Dataset):
    """Reader composition + joint index + a minimal Dataset surface."""

    #: Modalities this dataset family understands. Also the reader-iteration
    #: order for the joint index (subclasses may pick a different
    #: ``_canonical_reader`` precedence; close order is irrelevant).
    VALID_MODALITIES = ('edep', 'sensor', 'hits', 'labl')

    def __init__(self, *, split, data_root, transform=None,
                 ignore_index=-1, loop=1):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.ignore_index = ignore_index
        self.loop = loop
        # Detector datasets have no test-time-augmentation path; kept as a
        # plain attr only so any generic consumer that probes it sees False.
        self.test_mode = False
        self.data_list = self.get_data_list()
        log.info(
            "Totally %d x %d samples in %s %s set.",
            len(self.data_list), self.loop,
            os.path.basename(self.data_root), split)

    # -- Dataset surface ---------------------------------------------------

    def prepare_train_data(self, idx):
        return self.transform(self.get_data(idx))

    def __getitem__(self, idx):
        return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    # -- modality wiring ----------------------------------------------------

    def _validate_modalities(self, modalities):
        mods = set(modalities)
        if not mods:
            raise ValueError("modalities is empty; must load at least one")
        unknown = mods - set(self.VALID_MODALITIES)
        if unknown:
            raise ValueError(
                f"Unknown modalities {unknown}; valid: {self.VALID_MODALITIES}")
        if mods == {'labl'}:
            raise ValueError(
                "Invalid modality combination ('labl',): labl is a dimension "
                "table and requires an instance-bearing modality ('edep' or "
                "'hits') to join against.")
        if mods == {'sensor', 'labl'}:
            raise ValueError(
                "Invalid modality combination ('sensor', 'labl'): sensor has "
                "no instance/particle separation, so labl can't be attached. "
                "Add 'hits' or 'edep' to the modalities tuple.")

    def _modality_root(self, modality):
        mod_dir = os.path.join(self._source_data_root, modality)
        return mod_dir if os.path.isdir(mod_dir) else self._source_data_root

    def _readers_named(self):
        """``(modality, reader)`` for each loaded modality, in
        ``VALID_MODALITIES`` order (the order the joint index expects)."""
        return [(m, getattr(self, f'{m}_reader'))
                for m in self.VALID_MODALITIES
                if getattr(self, f'{m}_reader', None) is not None]

    def _build_joint_index(self, source_label, filter_label=''):
        """Build one joint cross-modality event index, injected into every
        reader so a global idx maps to the SAME physics event in all
        modalities (Phase A / D42 — see :mod:`pimm_data._joint_index`)."""
        self._n_events = build_joint_index(
            self._readers_named(), strict_lengths=self._strict_lengths,
            source_label=source_label, filter_label=filter_label)

    def get_data_list(self):
        n = getattr(self, '_n_events', 0)
        max_len = getattr(self, '_max_len', -1)
        if max_len > 0:
            n = min(n, max_len)
        return list(range(n))

    def get_data_name(self, idx):
        reader = self._canonical_reader
        file_idx, event_num = reader.locate(idx)
        fname = os.path.basename(reader.h5_files[file_idx])
        return f"{fname}_evt{event_num:03d}"

    def __del__(self):
        for m in self.VALID_MODALITIES:
            reader = getattr(self, f'{m}_reader', None)
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass


#: Back-compat alias for the pre-consolidation name (internal; only
#: jaxtpc.py / lucid.py imported it).
_MultiModalShardDataset = ShardEventDataset

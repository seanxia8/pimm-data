"""Shared composition base for the multimodal shard datasets (consolidation).

:class:`~pimm_data.jaxtpc.JAXTPCDataset` and
:class:`~pimm_data.lucid.LUCiDDataset` both compose per-modality shard readers
(edep / sensor / hits / labl), build one joint cross-modality event index
(Phase A / D42), and expose the same :class:`DefaultDataset` surface
(``get_data_list`` / ``get_data_name`` / ``prepare_test_data`` / teardown). That
wiring lives here once; each subclass supplies only its reader construction
(``__init__``) and its per-modality ``_build_*`` cloud builders / label model.

A subclass must, in ``__init__`` (before ``super().__init__``): set
``_modalities`` / ``_max_len`` / ``_strict_lengths`` / ``_source_data_root``,
create the ``{edep,sensor,hits,labl}_reader`` attrs (None when absent), set
``_canonical_reader`` (the reader whose file names label events), and call
``self._build_joint_index(source_label, filter_label)``.
"""

import os
from copy import deepcopy

import numpy as np

from .defaults import DefaultDataset
from ._joint_index import build_joint_index


class _MultiModalShardDataset(DefaultDataset):
    """Reader composition + joint index + DefaultDataset surface."""

    #: Modalities this dataset family understands. Also the reader-iteration
    #: order for the joint index (subclasses may pick a different
    #: ``_canonical_reader`` precedence; close order is irrelevant).
    VALID_MODALITIES = ('edep', 'sensor', 'hits', 'labl')

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

    # -- DefaultDataset surface --------------------------------------------

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

    def prepare_test_data(self, idx):
        """Test-time prep. ``segment`` is produced at the top level by a
        terminal :class:`Collect` transform only when the task needs it, so the
        pop is conditional (the nested-output datasets don't always emit one)."""
        data_dict = self.transform(self.get_data(idx))
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
        result_dict["fragment_list"] = [self.post_transform(f)
                                        for f in fragment_list]
        return result_dict

    def __del__(self):
        for m in self.VALID_MODALITIES:
            reader = getattr(self, f'{m}_reader', None)
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass

"""
Optical PMT-light readers — per-chunk waveforms from goop light output.

goop emits two on-disk groupings, sharing the SAME per-chunk fields
(``adc / offsets / t0_ns / pmt_id`` + a ``pe_counts`` table):

* **label schema** (``goop/io.py`` writer; doraemon files) —
  ``event_NNN/label_K/{adc,offsets,t0_ns,pmt_id,pe_counts}`` (+ ``tpc_*``).
  Each ``label_K`` is one interaction/volume; the group id is the interaction.
* **east/west schema** (``helix/optical/io.py``; ``light_output.h5``) —
  ``event_NNN/{east,west}/{adc,offsets,t0_ns,pmt_id}`` + event-level
  ``pe_counts_{east,west}``. The group id is the side; ``pmt_id`` is per-side.

Both read identically into per-**chunk** rows — chunks are long dense waveforms
(label: 20–50M samples/event; east/west: ~6M), so the loader emits chunks as the
unit and packs the raw samples losslessly (REDESIGN §3, two row-spaces). The
only seam between schemas is :meth:`OpticalSensorReader._groups` (how an event's
chunk groups are enumerated) and, for east/west, file discovery.

Per-chunk fields (what every goop/helix consumer uses — no synthetic coord):

    pmt_id      (K,)   int32    — channel (per-side for east/west)
    t0_ns       (K,)   float32  — absolute chunk start time
    length      (K,)   int32    — samples per chunk
    pe          (K,)   int32    — pe_counts[group][pmt] (channel total in group,
                                  shared across that channel's chunks; -1 if absent)
    instance    (K,)   int32    — group id (interaction for label, side for e/w)
    adc         (ΣL,)  float32  — packed, pedestal-subtracted samples

``adc`` is the packed second row-space; the dataset/Collect tag it
``('instance','sensor_wave_offset')`` so collate concats it and ``split_event``
slices it by sample count.
"""

import glob
import os

import numpy as np

from .._shard_meta import read_shard_meta
from ._base import ShardReaderBase


class OpticalSensorReader(ShardReaderBase):
    """Per-chunk PMT light reader for the goop ``label_K`` schema.

    Parameters
    ----------
    data_root, split, dataset_name : str
        Shard location (``{dataset_name}_sensor_*.h5`` under ``data_root`` /
        ``split``), as for the other readers.
    decode_digitization : bool
        Subtract the file's ``/config`` pedestal from the uint16 ADC (default
        True) — chunks are stored digitized; downstream works in subtracted ADC.
    """

    _MODALITY = 'sensor'

    #: Human label for the per-chunk ``instance`` group (the goop ``label_key``).
    group_kind = 'label'

    def __init__(self, data_root, split='', dataset_name='optical',
                 decode_digitization=True, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.decode_digitization = decode_digitization
        # Readout geometry / digitization — cached from the first shard's config
        # (one read via the A1 cache, no extra open).
        self.tick_ns = None
        self.pedestal = 0.0
        self.gain = 1.0
        self.n_channels = None
        self.n_pmts_per_side = None
        self._init_shards()

    def _index_for_shard(self, h5_path):
        """Present events; also caches readout geometry from the config (A1)."""
        meta = read_shard_meta(h5_path)
        if self.tick_ns is None:
            ca = meta['config_attrs']
            self.tick_ns = float(ca.get('tick_ns', 1.0))
            self.pedestal = float(ca.get('pedestal', 0.0))
            self.gain = float(ca.get('gain', 1.0))
            self.n_channels = (int(ca['n_channels'])
                               if 'n_channels' in ca else None)
            self.n_pmts_per_side = (int(ca['n_pmts_per_side'])
                                    if 'n_pmts_per_side' in ca else None)
            if 'label_key' in ca:
                self.group_kind = str(ca['label_key'])
        return meta['present_events']

    def _groups(self, evt):
        """Yield ``(group_id, h5_group, pe_counts|None)`` per chunk group.

        Label schema: one ``label_K`` group per interaction, sorted by id, with
        its own per-channel ``pe_counts``.
        """
        for lk in sorted((k for k in evt.keys() if k.startswith('label_')),
                         key=lambda k: int(k.split('_', 1)[1])):
            g = evt[lk]
            pe = g['pe_counts'][:] if 'pe_counts' in g else None
            yield int(lk.split('_', 1)[1]), g, pe

    def read_event(self, idx):
        """One event → flat dict of per-chunk arrays + packed ``adc``.

        Chunks are gathered over every group (:meth:`_groups`), in group order;
        within a group they stay in CSR (offset) order.
        """
        f, event_key = self._locate_event(idx)
        evt = f[event_key]
        ped = self.pedestal if self.decode_digitization else 0.0

        pmt, t0, length, pe, grp, adc_parts = [], [], [], [], [], []
        for gid, g, pe_counts in self._groups(evt):
            pmt_id = g['pmt_id'][:].astype(np.int32)
            if pmt_id.size == 0:
                continue
            offs = g['offsets'][:]
            a = g['adc'][:].astype(np.float32)
            if ped:
                a -= ped
            # offsets are CSR into adc; slice the covered span (tolerates a
            # leading/trailing gap), per-chunk lengths from the diff.
            adc_parts.append(a[int(offs[0]):int(offs[-1])])
            length.append(np.diff(offs).astype(np.int32))
            pmt.append(pmt_id)
            t0.append(g['t0_ns'][:].astype(np.float32))
            pe.append(pe_counts[pmt_id].astype(np.int32) if pe_counts is not None
                      else np.full(pmt_id.shape[0], -1, np.int32))
            grp.append(np.full(pmt_id.shape[0], gid, np.int32))

        if not pmt:                       # empty event (no groups/chunks)
            z = lambda dt: np.empty(0, dt)
            return {'pmt_id': z(np.int32), 't0_ns': z(np.float32),
                    'length': z(np.int32), 'pe': z(np.int32),
                    'instance': z(np.int32), 'adc': z(np.float32)}
        return {
            'pmt_id': np.concatenate(pmt),
            't0_ns': np.concatenate(t0),
            'length': np.concatenate(length),
            'pe': np.concatenate(pe),
            'instance': np.concatenate(grp),
            'adc': np.concatenate(adc_parts),
        }


class OpticalEastWestReader(OpticalSensorReader):
    """Per-chunk PMT light reader for the east/west schema (``light_output.h5``).

    Same per-chunk emission as the label reader; the group is the **side**
    (``instance`` = 0 east / 1 west), ``pmt_id`` is per-side, and ``pe_counts``
    is the event-level ``pe_counts_{side}`` table. These files are not
    shard-named (``light_output.h5``), so discovery globs ``*.h5``.
    """

    group_kind = 'side'
    SIDES = ('east', 'west')

    def _find_files(self):
        """Glob any ``*.h5`` under ``data_root`` (/ ``split``) — east/west files
        are not ``{name}_sensor_*.h5`` shards."""
        for pattern in (os.path.join(self.data_root, self.split, '*.h5'),
                        os.path.join(self.data_root, '*.h5')):
            files = sorted(glob.glob(pattern))
            if files:
                return files
        return []

    def _groups(self, evt):
        """Yield ``(side_idx, side_group, pe_counts_{side})`` for each side."""
        for sidx, side in enumerate(self.SIDES):
            if side not in evt:
                continue
            pek = f'pe_counts_{side}'
            pe = evt[pek][:] if pek in evt else None
            yield sidx, evt[side], pe

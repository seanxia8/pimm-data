"""
OpticalSensorReader — reads per-interaction PMT light waveforms from the
doraemon optical ``sensor/`` HDF5 files (``label_N`` schema).

These shards follow pimm-data's naming (``{dataset_name}_sensor_*.h5``) and
carry the standard ``/config`` group (``n_channels``, ``pedestal``, ``tick_ns``,
``global_event_offset``, ``file_index``, ``n_events``), so the
:class:`~pimm_data.readers._base.ShardReaderBase` index/locate/fork-safe
lifecycle is reused unchanged.

Schema (verified 2026-06-13)::

    event_NNN/label_K/
        adc        (ΣL,)        uint16  — concatenated chunk samples (digitized)
        offsets    (n_chunks+1,) int64  — CSR slice into adc
        pmt_id     (n_chunks,)  int32   — global channel 0..n_channels-1
        t0_ns      (n_chunks,)  float32 — chunk start time
        pe_counts  (n_channels,) int32  — per-(label, channel) true PE
        tpc_*      …                    — per-interaction TPC truth (NOT read here)

Each ``label_K`` group is one interaction; its chunks are the
*per-interaction* light contributions, so the same PMT can hold
time-overlapping chunks under different interactions (this is the
operator-discrimination signal, not a summed readout). The reader keeps that
structure: one **chunk** is one row, tagged with its interaction (``label_K``).

The chunks are long, dense waveforms (~36k samples each; ~1.4–3.4k chunks /
event → 20–50M samples). A per-sample point cloud is infeasible, so the reader
emits chunks as the unit and packs the raw samples losslessly:

    pmt_id      (K,)   int32    — per-chunk channel
    t0_ns       (K,)   float32  — per-chunk start time
    length      (K,)   int32    — samples in each chunk
    pe          (K,)   int32    — per-chunk true PE (pe_counts[label][pmt_id])
    interaction (K,)   int32    — interaction (label) index
    adc         (ΣL,)  float32  — packed, pedestal-subtracted samples

``adc`` is a SECOND row-space (samples), keyed by per-chunk ``length``; the
dataset/Collect tag it ``('instance', 'sensor_wave_offset')`` so collate concats
it and ``split_event`` slices it by sample count (REDESIGN §3, two row-spaces).
"""

import numpy as np

from .._shard_meta import read_shard_meta
from ._base import ShardReaderBase


class OpticalSensorReader(ShardReaderBase):
    """Reads per-interaction PMT light chunks from optical ``sensor/`` files.

    Parameters
    ----------
    data_root : str
        Directory containing the optical sensor shard files.
    split : str
        Split name (used as subdirectory when present).
    dataset_name : str
        File prefix — matches ``{dataset_name}_sensor_*.h5``.
    decode_digitization : bool
        Subtract the file's ``/config`` pedestal from the uint16 ADC (default
        True). The chunks are stored digitized; downstream noise/recon work in
        pedestal-subtracted ADC space.
    """

    _MODALITY = 'sensor'

    def __init__(self, data_root, split='', dataset_name='optical',
                 decode_digitization=True, **kwargs):
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.decode_digitization = decode_digitization
        # Readout geometry / digitization — cached from the first shard's config
        # (one read via the A1 cache, no extra open). Authoritative for coord
        # tick conversion and the pedestal subtraction.
        self.tick_ns = None
        self.pedestal = 0.0
        self.n_channels = None
        self._init_shards()

    def _index_for_shard(self, h5_path):
        """Present events; also caches readout geometry from the config (A1)."""
        meta = read_shard_meta(h5_path)
        if self.tick_ns is None:
            ca = meta['config_attrs']
            self.tick_ns = float(ca.get('tick_ns', 1.0))
            self.pedestal = float(ca.get('pedestal', 0.0))
            self.n_channels = (int(ca['n_channels'])
                               if 'n_channels' in ca else None)
        return meta['present_events']

    def read_event(self, idx):
        """One event → flat dict of per-chunk arrays + packed ``adc``.

        Chunks are gathered across every ``label_K`` interaction group, in
        sorted label order; within a label they are in CSR (offset) order.
        """
        f, event_key = self._locate_event(idx)
        evt = f[event_key]
        ped = self.pedestal if self.decode_digitization else 0.0

        pmt, t0, length, pe, inter, adc_parts = [], [], [], [], [], []
        for lk in sorted(evt.keys()):
            if not lk.startswith('label_'):
                continue
            lab = int(lk.split('_', 1)[1])
            g = evt[lk]
            pmt_id = g['pmt_id'][:].astype(np.int32)
            if pmt_id.size == 0:
                continue
            offs = g['offsets'][:]
            a = g['adc'][:].astype(np.float32)
            if ped:
                a -= ped
            # offsets are CSR into adc; slice the covered span (tolerates a
            # leading/trailing gap) and recover per-chunk lengths from the diff.
            adc_parts.append(a[int(offs[0]):int(offs[-1])])
            length.append(np.diff(offs).astype(np.int32))
            pmt.append(pmt_id)
            t0.append(g['t0_ns'][:].astype(np.float32))
            pe.append(g['pe_counts'][:][pmt_id].astype(np.int32))  # per-chunk truth
            inter.append(np.full(pmt_id.shape[0], lab, dtype=np.int32))

        if not pmt:                       # empty event (no interactions/chunks)
            return {
                'pmt_id': np.empty(0, np.int32),
                't0_ns': np.empty(0, np.float32),
                'length': np.empty(0, np.int32),
                'pe': np.empty(0, np.int32),
                'interaction': np.empty(0, np.int32),
                'adc': np.empty(0, np.float32),
            }
        return {
            'pmt_id': np.concatenate(pmt),
            't0_ns': np.concatenate(t0),
            'length': np.concatenate(length),
            'pe': np.concatenate(pe),
            'interaction': np.concatenate(inter),
            'adc': np.concatenate(adc_parts),
        }

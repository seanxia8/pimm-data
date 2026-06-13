"""
OpticalDataset — PMT light-readout dataset for the doraemon optical
simulation output (``label_N`` chunk schema).

A single-modality sibling of :class:`~pimm_data.jaxtpc.JAXTPCDataset` /
:class:`~pimm_data.lucid.LUCiDDataset`: it composes one
:class:`~pimm_data.readers.optical_sensor.OpticalSensorReader` over sharded
``sensor/`` files and emits a **nested** dict with a single ``sensor`` modality::

    {
      'sensor': {
        'coord':    (K, 2)  float32  — [pmt_id, t0_tick] per chunk
        'pmt_id':   (K,)    int32
        't0_ns':    (K,)    float32
        'length':   (K,)    int32    — samples per chunk
        'pe':       (K, 1)  float32  — per-chunk true PE
        'instance': (K,)    int32    — interaction (label) index
        'adc':      (ΣL,)   float32  — packed pedestal-subtracted waveforms
        '_roles':   {'adc': ('instance', 'sensor_wave_offset')},
      },
      'name': str, 'split': str,
    }

The optical readout is **per-interaction** (each ``label_K`` is one interaction;
the same PMT can carry overlapping chunks under different interactions). Each
**chunk** is one row of the ``sensor`` part (``sensor_offset`` counts chunks /
event); the raw samples are a packed SECOND row-space (``sensor_wave_offset``
counts samples / event), tagged ``('instance', 'sensor_wave_offset')`` so
collate concats them and ``split_event`` slices them by sample count
(REDESIGN §3, two row-spaces). ``instance`` is the interaction id — no ``labl``
join is needed; the per-interaction TPC truth (``tpc_*``) is left on disk for
now.

Canonical pipeline (note the second offset + the per-chunk waveform payload)::

    OpticalDataset(data_root=..., modalities=('sensor',), dataset_name='...',
      transform=[dict(type='Collect', modalities={
          'sensor': dict(
              keys=('coord', 'length', 'pe', 'instance', 'adc'),
              feat_keys=('pe',),
              offset_keys_dict=dict(offset='coord', wave_offset='adc'),
          )})])
    # `length` recovers per-chunk waveform boundaries from the packed `adc`.
    # collate_fn([ds[0], ds[1]]) -> sensor_coord/sensor_pe/sensor_instance/
    #   sensor_adc + sensor_offset (chunks) + sensor_wave_offset (samples) + _roles

Registered in :data:`pimm_data.DATASETS`.
"""

import numpy as np

from .builder import DATASETS
from ._dataset_base import ShardEventDataset
from .readers.optical_sensor import OpticalSensorReader


@DATASETS.register_module()
class OpticalDataset(ShardEventDataset):
    """PMT light dataset with per-interaction waveform chunks (one modality)."""

    #: Only the optical readout is loaded today (TPC truth stays on disk).
    VALID_MODALITIES = ('sensor',)

    def __init__(
        self,
        data_root,
        split='',
        modalities=('sensor',),
        dataset_name='optical',
        decode_digitization=True,
        transform=None,
        loop=1,
        max_len=-1,
        ignore_index=-1,
        strict_lengths=False,
    ):
        self._modalities = tuple(modalities)
        self._validate_modalities(self._modalities)

        self._dataset_name = dataset_name
        self._max_len = max_len
        self._strict_lengths = strict_lengths
        self._source_data_root = data_root
        self._source_split = split

        self.sensor_reader = None
        if 'sensor' in self._modalities:
            self.sensor_reader = OpticalSensorReader(
                data_root=self._modality_root('sensor'), split=split,
                dataset_name=dataset_name,
                decode_digitization=decode_digitization)

        self._canonical_reader = self.sensor_reader
        self._build_joint_index(source_label=f"OpticalDataset({data_root!r})")

        super().__init__(
            split=split, data_root=data_root,
            transform=transform, ignore_index=ignore_index, loop=loop,
        )

    def get_data(self, idx):
        real_idx = idx % len(self.data_list)
        data = {
            'name': self.get_data_name(real_idx),
            'split': self.split if isinstance(self.split, str) else 'custom',
        }
        if self.sensor_reader is not None:
            data['sensor'] = self._build_sensor(
                self.sensor_reader.read_event(real_idx))
        return data

    def _build_sensor(self, raw):
        """Per-chunk light cloud: coord=[pmt_id, t0_tick], packed ``adc`` as the
        second (sample) row-space; ``instance`` = interaction id."""
        pmt = raw['pmt_id']
        tick_ns = self.sensor_reader.tick_ns or 1.0
        t0_tick = np.round(raw['t0_ns'] / tick_ns).astype(np.float32)
        coord = np.stack([pmt.astype(np.float32), t0_tick], axis=1)  # (K, 2)
        return {
            'coord': coord,
            'pmt_id': pmt,
            't0_ns': raw['t0_ns'],
            'length': raw['length'],
            'pe': raw['pe'][:, None].astype(np.float32),
            'instance': raw['interaction'],
            'adc': raw['adc'],
            # adc is the packed sample row-space (REDESIGN §3); stamped here so
            # the Collect config only needs keys= + the wave_offset entry.
            '_roles': {'adc': ('instance', 'sensor_wave_offset')},
        }

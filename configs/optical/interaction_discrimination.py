"""Data-loader recipe — Optical interaction/operator discrimination (label schema).

CAMPAIGN.md row: Optical | Interaction/operator discrimination | schema='label',
modalities=('sensor',). Each `sensor` row is a per-interaction waveform CHUNK;
`instance` = interaction (label_K) is the per-chunk target; the packed `adc` is the
second (sample) row-space, role ('instance','sensor_wave_offset').

Data-loading half only (new flat-prefixed API). The model is a PLACEHOLDER:
PT-v3 does not fit 1-D waveform chunks — a waveform/sequence model is needed (see
CAMPAIGN.md; the research `coeff_foundation_model` is the relevant direction).

Expected batch after collate_fn:
    sensor_pmt_id (K,)  sensor_t0_ns (K,)  sensor_length (K,)  sensor_pe (K,1)
    sensor_instance (K,)  sensor_adc (ΣL,)  sensor_feat (K,1)=[pe]
    sensor_offset (B,)=chunks/event   sensor_wave_offset (B,)=samples/event
    name  split  _roles{sensor_adc: ('instance','sensor_wave_offset')}
"""
import os

# label schema (doraemon): /sdf/data/neutrino/doraemon/optical_test_00_00_02,
# dataset_name='test_00_00_02_pixel'. east/west (light_output.h5): schema='east_west'.
_data_root = os.environ.get("OPTICAL_DATA_ROOT",
                            "/sdf/data/neutrino/doraemon/optical_test_00_00_02")
_dataset_name = os.environ.get("OPTICAL_DATASET_NAME", "test_00_00_02_pixel")

_collect = dict(type="Collect", parts={
    "sensor": dict(
        keys=("pmt_id", "t0_ns", "length", "pe", "instance", "adc"),
        feat_keys=("pe",),
        offset_keys_dict=dict(offset="pmt_id", wave_offset="adc"),
    )})

# No geometric Apply: waveform chunks are not point clouds. Per-chunk preprocessing
# (e.g. baseline/scale, wavelet transform) would be added as sensor-scoped transforms.
train_transform = [_collect]
test_transform = [_collect]

data = dict(
    num_classes=None,                  # TODO: number of interaction/operator classes
    ignore_index=-1,
    names=None,
    train=dict(type="OpticalDataset", data_root=_data_root, split="",
               dataset_name=_dataset_name, schema="label", modalities=("sensor",),
               transform=train_transform, max_len=-1),
    val=dict(type="OpticalDataset", data_root=_data_root, split="",
             dataset_name=_dataset_name, schema="label", modalities=("sensor",),
             transform=test_transform, max_len=1000),
)

# model = dict(...)  # PLACEHOLDER — needs a waveform/sequence model, not PT-v3.

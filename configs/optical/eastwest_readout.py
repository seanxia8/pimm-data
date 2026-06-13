"""Data-loader recipe — Optical east/west readout (light_output.h5 schema).

CAMPAIGN.md (optical): covers the east/west-schema challenges — side-aware
readout, waveform SSL, and denoising/compression — which all load the SAME
per-chunk waveforms; they differ only in the (model-side) target. `instance` =
side (0=east, 1=west). Data-loading half only; PT-v3 does not fit waveforms — a
waveform/sequence model is needed (placeholder).

Batch: sensor_pmt_id/t0_ns/length/pe/instance(side)/adc + sensor_offset(chunks)
+ sensor_wave_offset(samples) + _roles{sensor_adc:('instance','sensor_wave_offset')}.
"""
import os

# east/west file (goop light_output.h5): point data_root at a dir whose sensor/
# holds the file(s) (globbed as *.h5; not shard-named).
_data_root = os.environ.get("OPTICAL_EASTWEST_DATA_ROOT", "/path/to/optical_eastwest")
_dataset_name = os.environ.get("OPTICAL_EASTWEST_DATASET_NAME", "light")

_collect = dict(type="Collect", modalities={
    "sensor": dict(
        keys=("pmt_id", "t0_ns", "length", "pe", "instance", "adc"),
        feat_keys=("pe",),
        offset_keys_dict=dict(offset="pmt_id", wave_offset="adc"),
    )})

data = dict(
    num_classes=None, ignore_index=-1, names=None,
    train=dict(type="OpticalDataset", data_root=_data_root, split="",
               dataset_name=_dataset_name, schema="east_west",
               modalities=("sensor",), transform=[_collect], max_len=-1),
    val=dict(type="OpticalDataset", data_root=_data_root, split="",
             dataset_name=_dataset_name, schema="east_west",
             modalities=("sensor",), transform=[_collect], max_len=1000),
)
# model = dict(...)  # PLACEHOLDER — needs a waveform/sequence model, not PT-v3.

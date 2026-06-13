"""Data-loader recipe — LUCiD sensor → segment reconstruction (vertex/energy/dir).

CAMPAIGN.md: LUCiD | sensor → seg recon | modalities=('sensor','step'). Loads BOTH
the PMT readout (input) and the 3D segments (target cloud) in one nested sample;
the model regresses event-level quantities from sensor against the step truth.
Data-loading half only; the exact regression target framing is model-side.

The same two-modality shape covers `sensor → inst denoising/deconvolution`
(swap `step` → `hits`).

Batch: sensor_coord/feat + step_coord/feat, each with its own offset.
"""
import os

_data_root = os.environ.get("LUCID_DATA_ROOT", "/path/to/wc")
_scale = 18.1

transform = [
    dict(type="AggregateSensorHits", modality="sensor",
         time_aggregation="earliest", flatten=False),
    dict(type="Apply", on="sensor", transforms=[
        dict(type="NormalizeCoord", center=[0, 0, 0], scale=_scale),
        dict(type="LogTransform", min_val=0.01, max_val=50.0, keys=("energy",)),
    ]),
    dict(type="Apply", on="step", transforms=[
        dict(type="NormalizeCoord", center=[0, 0, 0], scale=_scale),
        dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
    ]),
    dict(type="Collect", modalities={
        "sensor": dict(keys=("coord",), feat_keys=("coord", "energy", "time")),
        "step": dict(keys=("coord",), feat_keys=("coord", "energy")),
    }),
]
data = dict(
    train=dict(type="LUCiDDataset", data_root=_data_root, split="",
               dataset_name="wc", modalities=("sensor", "step"),
               transform=transform, max_len=-1),
    val=dict(type="LUCiDDataset", data_root=_data_root, split="",
             dataset_name="wc", modalities=("sensor", "step"),
             transform=transform, max_len=1000),
)
# model = dict(...)  # placeholder (sensor encoder -> regression/recon head).

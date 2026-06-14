"""Data-loader recipe — LUCiD raw-PMT self-supervised pretraining (sensor).

CAMPAIGN.md: LUCiD | SSL on raw PMT readout | modalities=('sensor',), no labels.
The new-API port of the existing sonata pretrain: AggregateSensorHits(flatten=
False) keeps the per-PMT aggregate NESTED (so MultiCrop(on='sensor') can read the
sub-dict), then MultiCrop replaces the old MultiViewGenerator. Data-loading half
only; model is a placeholder (Sonata + PT-v3).
"""
import os

_data_root = os.environ.get("LUCID_DATA_ROOT", "/path/to/wc")
_scale = 18.1                          # m
_view_aug = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
]
transform = [
    dict(type="AggregateSensorHits", modality="sensor",
         time_aggregation="earliest", flatten=False),
    dict(type="Apply", on="sensor", transforms=[
        dict(type="NormalizeCoord", center=[0, 0, 0], scale=_scale),
        dict(type="LogTransform", min_val=0.01, max_val=50.0, keys=("energy",)),
        dict(type="RelativeLogNormalize", scale=50.0, max_val=4000.0,
             out_min=-1.0, out_max=1.0, keys=("time",)),
        dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    ]),
    dict(type="MultiCrop", on="sensor",
         view_keys=("coord", "origin_coord", "energy", "time"),
         global_view_num=2, global_view_scale=(0.55, 1.0),
         local_view_num=6, local_view_scale=(0.15, 0.45),
         global_transform=_view_aug, local_transform=_view_aug, max_size=30000),
    dict(type="Collect", parts={
        "global": dict(keys=("coord", "origin_coord", "energy", "time", "offset"),
                       offset_keys_dict={}, feat_keys=("coord", "energy", "time")),
        "local": dict(keys=("coord", "energy", "time", "offset"),
                      offset_keys_dict={}, feat_keys=("coord", "energy", "time")),
    }),
]
data = dict(
    train=dict(type="LUCiDDataset", data_root=_data_root, split="",
               dataset_name="wc", modalities=("sensor",), transform=transform,
               max_len=-1),
    val=dict(type="LUCiDDataset", data_root=_data_root, split="",
             dataset_name="wc", modalities=("sensor",), transform=transform,
             max_len=1000),
)
# model = dict(...)  # placeholder (Sonata + PT-v3).

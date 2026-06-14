"""Data-loader recipe — LUCiD SSL on per-particle PMT decomposition (hits).

CAMPAIGN.md: LUCiD | SSL on per-particle PMT decomposition | modalities=('hits',).
Multi-crop SSL on the per-particle hit point cloud. Data-loading half only.
"""
import os

_data_root = os.environ.get("LUCID_DATA_ROOT", "/path/to/wc")
_scale = 18.1
_view_aug = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
]
transform = [
    dict(type="Apply", on="hits", transforms=[
        dict(type="NormalizeCoord", center=[0, 0, 0], scale=_scale),
        dict(type="LogTransform", min_val=0.01, max_val=50.0, keys=("energy",)),
        dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    ]),
    dict(type="MultiCrop", on="hits", view_keys=("coord", "origin_coord", "energy"),
         global_view_num=2, global_view_scale=(0.55, 1.0),
         local_view_num=6, local_view_scale=(0.15, 0.45),
         global_transform=_view_aug, local_transform=_view_aug, max_size=30000),
    dict(type="Collect", parts={
        "global": dict(keys=("coord", "origin_coord", "energy", "offset"),
                       offset_keys_dict={}, feat_keys=("coord", "energy")),
        "local": dict(keys=("coord", "energy", "offset"),
                      offset_keys_dict={}, feat_keys=("coord", "energy")),
    }),
]
data = dict(
    train=dict(type="LUCiDDataset", data_root=_data_root, split="",
               dataset_name="wc", modalities=("hits",), transform=transform,
               max_len=-1),
    val=dict(type="LUCiDDataset", data_root=_data_root, split="",
             dataset_name="wc", modalities=("hits",), transform=transform,
             max_len=1000),
)
# model = dict(...)  # placeholder (SSL backbone).

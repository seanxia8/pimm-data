"""Data-loader recipe — JAXTPC 3D self-supervised pretraining (segments).

CAMPAIGN.md: JAXTPC | 3D self-supervised (SSL) | modalities=('step',), no labels.
Multi-crop SSL: MultiCrop packs global/local views; the model masks/contrasts.
Data-loading half only (new flat-prefixed API); model is a placeholder.

Batch: global_coord/origin_coord/energy/feat/offset, local_coord/energy/feat/offset.
"""
import os

_data_root = os.environ.get("JAXTPC_DATA_ROOT", "/path/to/jaxtpc/production")
_scale = 2160.0 * 3 ** 0.5
_view_aug = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
]
transform = [
    dict(type="Apply", on="step", transforms=[
        dict(type="NormalizeCoord", center=[0, 0, 0], scale=_scale),
        dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
        dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    ]),
    dict(type="MultiCrop", on="step", view_keys=("coord", "origin_coord", "energy"),
         global_view_num=2, global_view_scale=(0.55, 1.0),
         local_view_num=6, local_view_scale=(0.15, 0.45),
         global_transform=_view_aug, local_transform=_view_aug, max_size=30000),
    dict(type="Collect", modalities={
        "global": dict(keys=("coord", "origin_coord", "energy", "offset"),
                       offset_keys_dict={}, feat_keys=("coord", "energy")),
        "local": dict(keys=("coord", "energy", "offset"),
                      offset_keys_dict={}, feat_keys=("coord", "energy")),
    }),
]
data = dict(
    train=dict(type="JAXTPCDataset", data_root=_data_root, split="train",
               dataset_name="sim", modalities=("step",), transform=transform,
               min_deposits=1024, max_len=-1),
    val=dict(type="JAXTPCDataset", data_root=_data_root, split="val",
             dataset_name="sim", modalities=("step",), transform=transform,
             min_deposits=1024, max_len=1000),
)
# model = dict(...)  # placeholder (SSL backbone, e.g. Sonata + PT-v3).

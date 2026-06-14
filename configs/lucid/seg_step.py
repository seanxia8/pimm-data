"""Data-loader recipe — LUCiD 3D semantic/instance segmentation on segments.

CAMPAIGN.md: LUCiD | 3D semantic/instance seg on segments | modalities=('step',),
labels=True (segment=particle category, instance=particle_idx). Data-loading half
only; model is a placeholder.

Batch: step_coord/grid_coord/segment/instance/feat(coord+energy)/offset.
"""
import os

_data_root = os.environ.get("LUCID_DATA_ROOT", "/path/to/wc")
_scale = 18.1
grid_size = 0.04

_geom = [
    dict(type="NormalizeCoord", center=[0, 0, 0], scale=_scale),
    dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
         return_grid_coord=True),
]
_aug = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
]
_collect = dict(type="Collect", parts={
    "step": dict(keys=("coord", "grid_coord", "segment", "instance"),
                 feat_keys=("coord", "energy"))})

train_transform = [dict(type="Apply", on="step", transforms=_geom + _aug), _collect]
test_transform = [dict(type="Apply", on="step", transforms=_geom), _collect]

data = dict(
    num_classes=None,                  # TODO: particle-category scheme
    ignore_index=-1, names=None,
    train=dict(type="LUCiDDataset", data_root=_data_root, split="",
               dataset_name="wc", modalities=("step",), labels=True,
               transform=train_transform, max_len=-1),
    val=dict(type="LUCiDDataset", data_root=_data_root, split="",
             dataset_name="wc", modalities=("step",), labels=True,
             transform=test_transform, max_len=1000),
)
# model = dict(...)  # placeholder.

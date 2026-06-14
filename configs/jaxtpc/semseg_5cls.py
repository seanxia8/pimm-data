"""Data-loader recipe — JAXTPC 3D semantic segmentation (motif 5-class).

CAMPAIGN.md row: JAXTPC | 3D semantic seg | modalities=('step',), labels='pdg'.

This file is the **data-loading half** (new flat-prefixed API: Apply(on=) +
Collect(modalities=)). The model/optimizer/hooks half is a placeholder to fill
when lifted into a pimm training config (see CAMPAIGN.md — DefaultSegmentorV2 +
PT-v3m2, in_channels=4).

Expected batch after collate_fn([ds[i], ...]):
    step_coord (N,3)  step_grid_coord (N,3)  step_segment (N,)
    step_feat (N,4)=[coord|energy]  step_offset (B,)  name  split  _roles
"""
import os

_data_root = os.environ.get("JAXTPC_DATA_ROOT", "/path/to/jaxtpc/production")
_center = [0.0, 0.0, 0.0]
_scale = 2160.0 * 3 ** 0.5            # ~3741 mm -> roughly [-1, 1]
grid_size = 0.001

_geom = [
    dict(type="NormalizeCoord", center=_center, scale=_scale),
    dict(type="LogTransform", min_val=0.01, max_val=20.0),
    dict(type="RemapSegment", scheme="motif_5cls"),   # raw pdg -> 5 dense classes
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
         mode="train", return_grid_coord=True),
]
_aug = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="x", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1, 1], axis="y", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
]
_collect = dict(type="Collect", parts={
    "step": dict(keys=("coord", "grid_coord", "segment"),
                 feat_keys=("coord", "energy"))})

train_transform = [dict(type="Apply", on="step", transforms=_geom + _aug), _collect]
test_transform = [dict(type="Apply", on="step", transforms=_geom), _collect]

data = dict(
    num_classes=5,
    ignore_index=-1,
    names=["shower", "track", "michel", "delta", "led"],
    train=dict(type="JAXTPCDataset", data_root=_data_root, split="train",
               dataset_name="sim", modalities=("step",), labels="pdg",
               transform=train_transform, min_deposits=1024, max_len=-1),
    val=dict(type="JAXTPCDataset", data_root=_data_root, split="val",
             dataset_name="sim", modalities=("step",), labels="pdg",
             transform=test_transform, min_deposits=1024, max_len=1000),
)

# model = dict(...)  # placeholder — lift into a pimm training config (CAMPAIGN.md).

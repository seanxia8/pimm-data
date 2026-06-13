"""Data-loader recipe — LUCiD per-PMT semantic + instance segmentation on hits.

CAMPAIGN.md row: LUCiD | Per-PMT semantic/instance seg | modalities=('hits',),
labels=True (inst=hits in the source doc). `segment` = labl.particle.category,
`instance` = particle_idx, attached by the LUCiD label decoration.

Data-loading half only (new flat-prefixed API). Model half is a placeholder.

Expected batch after collate_fn:
    hits_coord (H,3)  hits_segment (H,)  hits_instance (H,)
    hits_feat (H,5)=[coord|energy|time]  hits_offset (B,)  name  split  _roles
"""
import os

_data_root = os.environ.get("LUCID_DATA_ROOT", "/path/to/wc")
_center = [0.0, 0.0, 0.0]
_scale = 18.1                          # m (SK-like detector half-extent)
grid_size = 0.04

_geom = [
    dict(type="NormalizeCoord", center=_center, scale=_scale),
    dict(type="LogTransform", min_val=0.01, max_val=50.0, keys=("energy",)),
    dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train",
         return_grid_coord=True, sum_keys=("energy",), min_keys=("time",)),
]
_aug = [
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomFlip", p=0.5),
]
_collect = dict(type="Collect", modalities={
    "hits": dict(keys=("coord", "grid_coord", "segment", "instance"),
                 feat_keys=("coord", "energy", "time"))})

train_transform = [dict(type="Apply", on="hits", transforms=_geom + _aug), _collect]
test_transform = [dict(type="Apply", on="hits", transforms=_geom), _collect]

data = dict(
    num_classes=None,                  # TODO: set from labl.particle.category scheme
    ignore_index=-1,
    names=None,                        # TODO: category names
    train=dict(type="LUCiDDataset", data_root=_data_root, split="",
               dataset_name="wc", modalities=("hits",), labels=True,
               transform=train_transform, pe_threshold=0.0, max_len=-1),
    val=dict(type="LUCiDDataset", data_root=_data_root, split="",
             dataset_name="wc", modalities=("hits",), labels=True,
             transform=test_transform, pe_threshold=0.0, max_len=1000),
)

# model = dict(...)  # placeholder — lift into a pimm training config (CAMPAIGN.md).

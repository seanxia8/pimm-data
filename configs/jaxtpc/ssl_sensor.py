"""Data-loader recipe — JAXTPC raw-readout self-supervised pretraining (sensor).

CAMPAIGN.md: JAXTPC | Raw-readout SSL | modalities=('sensor',), no labels.
Multi-crop SSL on the sparse wire/pixel readout point cloud. NOTE: sensor `coord`
is detector-index space — 2D (wire, time) for wire readout, 3D for pixel
(auto-detected) — so coord NORMALIZATION is readout-dependent and left to the
lift-into-pimm step; MultiCrop's distance-based cropping is scale-invariant, so
it operates on raw coord here. Data-loading half only; model is a placeholder.
"""
import os

_data_root = os.environ.get("JAXTPC_DATA_ROOT", "/path/to/jaxtpc/production")
# No geometric view-aug: sensor coord is detector-index space (2D/3D), where
# rotations/flips are not meaningful — leave per-readout aug to the pimm lift.
transform = [
    dict(type="Apply", on="sensor", transforms=[
        dict(type="LogTransform", min_val=0.01, max_val=20.0, keys=("energy",)),
        dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    ]),
    dict(type="MultiCrop", on="sensor", view_keys=("coord", "origin_coord", "energy"),
         global_view_num=2, global_view_scale=(0.55, 1.0),
         local_view_num=6, local_view_scale=(0.15, 0.45), max_size=30000),
    dict(type="Collect", modalities={
        "global": dict(keys=("coord", "origin_coord", "energy", "offset"),
                       offset_keys_dict={}, feat_keys=("coord", "energy")),
        "local": dict(keys=("coord", "energy", "offset"),
                      offset_keys_dict={}, feat_keys=("coord", "energy")),
    }),
]
data = dict(
    train=dict(type="JAXTPCDataset", data_root=_data_root, split="train",
               dataset_name="sim", modalities=("sensor",), transform=transform,
               max_len=-1),
    val=dict(type="JAXTPCDataset", data_root=_data_root, split="val",
             dataset_name="sim", modalities=("sensor",), transform=transform,
             max_len=1000),
)
# model = dict(...)  # placeholder (SSL backbone).

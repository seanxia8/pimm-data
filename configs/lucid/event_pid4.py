"""Data-loader recipe — LUCiD event classification: 4-class single-particle PID.

CAMPAIGN.md: LUCiD | Event class/regression | MultiModalEventDataset over four
single-particle WAND configs (mu- / e- / pi+ / pi0). Data-loading half only;
model is a placeholder (pool sensor cloud -> 4-class head vs event_label).
"""
import os

_data_root = os.environ.get("WAND_DATA_ROOT",
                            "/sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like")
_scale = 18.1
grid_size = 0.04

_sources = [
    dict(name="config_000001", label=0, config_id=0),   # mu-
    dict(name="config_000003", label=1, config_id=1),   # e-
    dict(name="config_000002", label=2, config_id=2),   # pi+
    dict(name="config_000005", label=3, config_id=3),   # pi0
]
_source_dataset = dict(type="LUCiDDataset", modalities=("sensor",), dataset_name="wc")
_holdout = dict(seed=0, n_per_config=2000)

_transform = [
    dict(type="AggregateSensorHits", modality="sensor",
         time_aggregation="earliest", flatten=False),
    dict(type="Apply", on="sensor", transforms=[
        dict(type="NormalizeCoord", center=[0, 0, 0], scale=_scale),
        dict(type="LogTransform", min_val=0.01, max_val=50.0, keys=("energy",)),
        dict(type="RelativeLogNormalize", scale=50.0, max_val=4000.0,
             out_min=-1.0, out_max=1.0, keys=("time",)),
        dict(type="GridSample", grid_size=grid_size, hash_type="fnv",
             mode="train", sum_keys=("energy",), min_keys=("time",)),
    ]),
    dict(type="Collect", modalities={
        "sensor": dict(keys=("coord",), feat_keys=("coord", "energy", "time"))}),
]


def _mmed(split, cap=-1):
    return dict(type="MultiModalEventDataset", source_dataset=_source_dataset,
                sources=_sources, data_root=_data_root, split=split,
                holdout=_holdout, min_points=1024, transform=_transform,
                max_events_per_source=cap)


data = dict(
    num_classes=4, names=["mu-", "e-", "pi+", "pi0"], ignore_index=-1,
    train=_mmed("train"), val=_mmed("val"), test=_mmed("test"),
)
# model = dict(...)  # placeholder.

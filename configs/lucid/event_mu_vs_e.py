"""Data-loader recipe — LUCiD event classification: mu- vs e- (single-ring PID).

CAMPAIGN.md: LUCiD | Event class/regression | MultiModalEventDataset over two
single-particle WAND configs with a deterministic per-config holdout. The
canonical Cherenkov lepton-ID task (mu- = sharp ring, e- = fuzzy ring).

`event_label` (the per-source label) is the target, carried to the batch as an
event-role scalar; the model pools the per-event sensor cloud and classifies.
Data-loading half only; model is a placeholder.
"""
import os

_data_root = os.environ.get("WAND_DATA_ROOT",
                            "/sdf/data/neutrino/cjesus/DORAEMON/WAND/SK_like")
_scale = 18.1
grid_size = 0.04

# source combination: which WAND config -> which class
_sources = [
    dict(name="config_000001", label=0, config_id=0),   # single mu-
    dict(name="config_000003", label=1, config_id=1),   # single e-
]
_source_dataset = dict(type="LUCiDDataset", modalities=("sensor",), dataset_name="wc")
# deterministic train/val/test holdout: n_per_config events held out per source
# (by smallest (config_id, file_index, source_event_idx) hash) -> val+test.
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
    dict(type="Collect", parts={
        "sensor": dict(keys=("coord",), feat_keys=("coord", "energy", "time"))}),
]


def _mmed(split, cap=-1):
    return dict(type="MultiModalEventDataset", source_dataset=_source_dataset,
                sources=_sources, data_root=_data_root, split=split,
                holdout=_holdout, min_points=1024, transform=_transform,
                max_events_per_source=cap)


data = dict(
    num_classes=2, names=["mu-", "e-"], ignore_index=-1,
    train=_mmed("train"), val=_mmed("val"), test=_mmed("test"),
)
# model = dict(...)  # placeholder — pool sensor cloud -> 2-class head vs event_label.

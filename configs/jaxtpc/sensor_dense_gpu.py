"""Data-loader recipe — JAXTPC sensor DENSE path (densify + add noise on GPU).

CAMPAIGN.md (JAXTPC): the sensor readout is loaded SPARSE in the worker, moved to
the GPU, then densified to per-plane images and given fresh intrinsic noise
(JAXTPC production omits coherent noise by default, so it is added at load). This
is the only recipe that uses the post-collate dense ops; everything else is
sparse point clouds.

It has two halves that run in different places:

1. **Worker (per-event, in the DataLoader)** — `transform=` below. Load the sparse
   sensor COO and `Collect` it to flat `sensor_*`. Only sparse data crosses PCIe.

       batch (after collate_fn): sensor_wire/time/value/plane_gid (ΣM,)
                                 sensor_offset (B,)  name  split  _roles

2. **GPU (post-collate)** — `gpu_transforms` below, a *policy* block. The per-plane
   geometry (`n_wires`/`n_ticks`) is only known at runtime (`ds.plane_geometry()`),
   so the GPU chain can't be materialized in the config dict; the runner expands it:

       stages = build_sensor_gpu_stages(ds.plane_geometry(), **gpu_transforms)
       batch  = stages(batch)        # -> batch["sensor_dense"] = {plane_gid: (B, W, T)}

   which is the Compose `[ToDevice, BatchDensify, BatchAddIntrinsicNoise, BatchDigitize]`
   (dense ops are ordinary scope='sample' transforms; ToDevice is the device step;
   noise self-seeds from batch['name'] folded with base_seed/epoch/rank).

The model (placeholder) is a dense/CNN model over `sensor_dense`.

NB: the dense ops still use `modality='sensor'` (read the flat `sensor_*` keys) —
that param is a different transform's, not `Collect`'s (which is now `part=`).
"""
import os

# real wire data: /sdf/data/neutrino/omara/JAXTPC_Wire/test_00_00_02
# (split='run_0027575715', dataset_name='sim_wire')
_data_root = os.environ.get("JAXTPC_DATA_ROOT", "/path/to/jaxtpc/production")

# 1) WORKER: collect the sparse sensor COO (flat sensor_*). No densify here.
_collect = dict(type="Collect", parts={
    "sensor": dict(keys=("wire", "time", "value", "plane_gid"))})

data = dict(
    train=dict(type="JAXTPCDataset", data_root=_data_root, split="train",
               dataset_name="sim", modalities=("sensor",),
               transform=[_collect], max_len=-1,
               # wire_lengths_per_plane={...}   # only needed if incoherent=True
               ),
    val=dict(type="JAXTPCDataset", data_root=_data_root, split="val",
             dataset_name="sim", modalities=("sensor",),
             transform=[_collect], max_len=1000),
)

# 2) GPU (post-collate) policy. Expanded by the runner with the dataset geometry:
#    build_sensor_gpu_stages(ds.plane_geometry(), **gpu_transforms)
gpu_transforms = dict(
    modality="sensor",     # read/write the flat sensor_* keys (-> sensor_dense)
    device="cuda",         # ToDevice step; the grids are born on the GPU
    coherent=True,         # add the coherent noise JAXTPC omits (on-device torch)
    incoherent=False,      # True (+ wire_lengths_per_plane) to also add incoherent
    # coherent_numpy=False,  # True -> bit-exact/device-independent numpy oracle (slower)
    digitize=True,
    n_bits=12,
    base_seed=0,           # folded with event name/epoch/rank -> reproducible noise
)
# model = dict(...)  # placeholder — a dense/CNN model over sensor_dense.

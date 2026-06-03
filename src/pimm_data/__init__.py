"""pimm-data — multimodal detector dataset loaders + transform library.

Imports are ordered so that registering side-effects populate the
:data:`TRANSFORMS` and :data:`DATASETS` registries before anything
downstream looks them up.
"""

# Register blosc/zstd/lz4 HDF5 filters at import so production output
# written with those codecs (run_batch --codec, default blosc-zstd) is
# transparently readable. No-op if hdf5plugin isn't installed.
try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

# Registries + build function
from ._registry import Registry, build_from_cfg
from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS

# Core dataset bases (register DefaultDataset, ConcatDataset)
from .defaults import DefaultDataset, ConcatDataset

# Collate utilities
from .collate import collate_fn, point_collate_fn, inseg_collate_fn

# Dataset classes (register themselves)
from .jaxtpc import JAXTPCDataset
from .lucid import LUCiDDataset
from .pilarnet import PILArNetH5Dataset
from .multimodal import MultiModalEventDataset

# Detector-specific transforms (register PDGToSemantic)
from . import detector_transforms  # noqa: F401

__all__ = [
    "Registry",
    "build_from_cfg",
    "DATASETS",
    "build_dataset",
    "Compose",
    "TRANSFORMS",
    "DefaultDataset",
    "ConcatDataset",
    "collate_fn",
    "point_collate_fn",
    "inseg_collate_fn",
    "JAXTPCDataset",
    "LUCiDDataset",
    "PILArNetH5Dataset",
    "MultiModalEventDataset",
]

__version__ = "0.1.0"

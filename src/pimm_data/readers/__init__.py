"""Readers for detector HDF5 files.

Each reader is self-contained: given a data_root and split, it discovers
shard files, builds an event index, and exposes ``read_event(idx)``
returning a flat ``dict[str, np.ndarray]``.
"""

from .jaxtpc_edep import JAXTPCEdepReader
from .jaxtpc_sensor import JAXTPCSensorReader
from .jaxtpc_labl import JAXTPCLablReader
from .jaxtpc_hits import JAXTPCHitsReader
from .lucid_edep import LUCiDEdepReader
from .lucid_sensor import LUCiDSensorReader
from .lucid_hits import LUCiDHitsReader
from .lucid_labl import LUCiDLablReader

__all__ = [
    "JAXTPCEdepReader",
    "JAXTPCSensorReader",
    "JAXTPCLablReader",
    "JAXTPCHitsReader",
    "LUCiDEdepReader",
    "LUCiDSensorReader",
    "LUCiDHitsReader",
    "LUCiDLablReader",
]

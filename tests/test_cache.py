"""Tests for pimm_data.utils.cache.

Two layers:
1. Smoke tests of module/package wiring (always run).
2. Functional round-trip tests of ``shared_dict`` (require the optional
   ``SharedArray`` package and a writable ``/dev/shm``; skipped otherwise).

Functional tests use per-test UUID-suffixed names and clean up shared
memory on teardown so re-runs are safe.
"""

import os
import uuid
import pytest
import numpy as np

try:
    import SharedArray as _sa
    _SHM_AVAILABLE = os.path.isdir('/dev/shm') and os.access('/dev/shm', os.W_OK)
except ImportError:
    _sa = None
    _SHM_AVAILABLE = False

needs_shm = pytest.mark.skipif(
    not _SHM_AVAILABLE,
    reason="Requires SharedArray package and writable /dev/shm",
)


def _cleanup_shm(prefix):
    """Remove any shared arrays + .keys file matching the prefix."""
    if _sa is None:
        return
    for entry in _sa.list():
        n = entry.name.decode() if isinstance(entry.name, bytes) else entry.name
        if n.startswith(prefix):
            try:
                _sa.delete(n)
            except Exception:
                pass
    keys_file = f"/dev/shm/{prefix}.keys"
    if os.path.exists(keys_file):
        try:
            os.remove(keys_file)
        except Exception:
            pass


@pytest.fixture
def shm_name():
    """Unique name per test + teardown cleanup."""
    name = f"pimmdata_test_{uuid.uuid4().hex[:12]}"
    yield name
    _cleanup_shm(name)


# ───── smoke tests (always run) ─────

def test_cache_module_exports():
    from pimm_data.utils.cache import shared_dict, shared_array
    assert callable(shared_dict)
    assert callable(shared_array)


def test_utils_package_reexports():
    from pimm_data.utils import shared_dict as via_package
    from pimm_data.utils.cache import shared_dict
    assert via_package is shared_dict


# ───── functional round-trip tests (skip without SharedArray) ─────

@needs_shm
def test_roundtrip_single_array(shm_name):
    from pimm_data.utils.cache import shared_dict
    arr = np.arange(10, dtype=np.float32)
    shared_dict(shm_name, var={"coord": arr})

    retrieved = shared_dict(shm_name)
    assert set(retrieved.keys()) == {"coord"}
    np.testing.assert_array_equal(retrieved["coord"], arr)
    assert retrieved["coord"].dtype == arr.dtype


@needs_shm
def test_roundtrip_multiple_arrays(shm_name):
    from pimm_data.utils.cache import shared_dict
    rng = np.random.default_rng(42)
    original = {
        "coord": rng.random((100, 3)).astype(np.float32),
        "segment": rng.integers(0, 5, size=100).astype(np.int32),
        "energy": rng.random((100, 1)).astype(np.float32),
    }
    shared_dict(shm_name, var=original)

    retrieved = shared_dict(shm_name)
    assert set(retrieved.keys()) == set(original.keys())
    for k, v in original.items():
        np.testing.assert_array_equal(retrieved[k], v)
        assert retrieved[k].dtype == v.dtype
        assert retrieved[k].shape == v.shape


@needs_shm
def test_nonnumpy_values_filtered(shm_name):
    """shared_dict should store only ndarray values and skip others."""
    from pimm_data.utils.cache import shared_dict
    original = {
        "coord": np.zeros((3, 3), dtype=np.float32),
        "name": "event_001",  # str, filtered
        "idx": 42,  # int, filtered
    }
    shared_dict(shm_name, var=original)

    retrieved = shared_dict(shm_name)
    assert "coord" in retrieved
    assert "name" not in retrieved
    assert "idx" not in retrieved


@needs_shm
def test_different_names_do_not_collide():
    """Two shared_dicts under different names stay independent."""
    from pimm_data.utils.cache import shared_dict
    n1 = f"pimmdata_test_{uuid.uuid4().hex[:12]}"
    n2 = f"pimmdata_test_{uuid.uuid4().hex[:12]}"
    try:
        shared_dict(n1, var={"a": np.zeros(5, dtype=np.float32)})
        shared_dict(n2, var={"a": np.ones(5, dtype=np.float32)})
        r1 = shared_dict(n1)
        r2 = shared_dict(n2)
        np.testing.assert_array_equal(r1["a"], np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(r2["a"], np.ones(5, dtype=np.float32))
    finally:
        _cleanup_shm(n1)
        _cleanup_shm(n2)


@needs_shm
def test_populate_then_retrieve_repeatedly(shm_name):
    """Simulates pimm's hook flow: one call populates, many later calls
    (as DataLoader workers would) attach and retrieve the same data."""
    from pimm_data.utils.cache import shared_dict
    arr = np.linspace(0, 1, 50, dtype=np.float32).reshape(10, 5)
    shared_dict(shm_name, var={"coord": arr})
    for _ in range(3):
        retrieved = shared_dict(shm_name)
        np.testing.assert_array_equal(retrieved["coord"], arr)


@needs_shm
def test_shared_array_attach_to_existing(shm_name):
    """Second shared_array call with the same name attaches to the
    existing segment instead of erroring (idempotent populate)."""
    from pimm_data.utils.cache import shared_array
    arr = np.arange(7, dtype=np.int64)
    a = shared_array(f"{shm_name}.k", var=arr)
    b = shared_array(f"{shm_name}.k", var=arr)
    np.testing.assert_array_equal(a, b)
"""Shared pytest fixtures.

Default: tests run against a synthetic v3 dataset written to a session-
scoped tmp dir by :mod:`pimm_data.testing`. No external data is needed —
a fresh clone can run the full suite end-to-end.

Override: point ``JAXTPC_DATA_ROOT`` / ``LUCID_DATA_ROOT`` at a real v3
dataset to run the same tests against production output. The fixture
validates the v3 layout (all of ``edep``/``sensor``/``hits``/``labl`` must
be present) and skips with a descriptive reason if the override is
incomplete.

Tests that only make sense on real data should use
``@pytest.mark.real_data_only`` — it auto-skips when we fall back to the
synthesizer.
"""

import os
import pytest

from pimm_data.testing import make_jaxtpc_sample, make_lucid_sample

_REQUIRED_SUBDIRS = ('edep', 'sensor', 'hits', 'labl')


def _validate_override(path):
    if not os.path.isdir(path):
        return None, f"root not found: {path}"
    missing = [d for d in _REQUIRED_SUBDIRS
               if not os.path.isdir(os.path.join(path, d))]
    if missing:
        return None, (f"{path} is not a v3 layout "
                      f"(missing: {', '.join(missing)})")
    return path, None


def _resolve_root(env_var, tmp_path_factory, builder, subdir):
    """Return (root_path, is_synthetic)."""
    override = os.environ.get(env_var)
    if override:
        path, reason = _validate_override(override)
        if path is None:
            pytest.skip(f"{env_var}={override} rejected ({reason})")
        return path, False
    root = str(tmp_path_factory.mktemp(subdir))
    builder(root)
    return root, True


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_data_only: skip when running against the synthetic fixture "
        "(i.e. no *_DATA_ROOT env var set).")


def pytest_collection_modifyitems(config, items):
    """Skip `real_data_only` tests when neither env var is set.

    Conservative: a test's mode is determined by which env var its
    fixture uses, but pytest doesn't know that at collection time.
    We apply the skip if *both* env vars are unset, which covers the
    fresh-clone / CI case where nothing is available.
    """
    if os.environ.get('JAXTPC_DATA_ROOT') or os.environ.get('LUCID_DATA_ROOT'):
        return
    skip_marker = pytest.mark.skip(
        reason="real_data_only: set JAXTPC_DATA_ROOT or LUCID_DATA_ROOT")
    for item in items:
        if 'real_data_only' in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope='session')
def jaxtpc_data_root(tmp_path_factory):
    path, _ = _resolve_root('JAXTPC_DATA_ROOT', tmp_path_factory,
                            make_jaxtpc_sample, 'jaxtpc_synth')
    return path


@pytest.fixture(scope='session')
def jaxtpc_pixel_data_root(tmp_path_factory):
    """Pixel-readout JAXTPC dataset.

    Prefers ``JAXTPC_PIXEL_DATA_ROOT`` when set; otherwise falls back
    to a synthetic pixel fixture. Pixel-specific tests should request
    this fixture instead of ``jaxtpc_data_root``.
    """
    from functools import partial
    builder = partial(make_jaxtpc_sample, readout_type='pixel')
    path, _ = _resolve_root('JAXTPC_PIXEL_DATA_ROOT', tmp_path_factory,
                            builder, 'jaxtpc_pixel_synth')
    return path


@pytest.fixture(scope='session')
def lucid_data_root(tmp_path_factory):
    path, _ = _resolve_root('LUCID_DATA_ROOT', tmp_path_factory,
                            make_lucid_sample, 'lucid_synth')
    return path


@pytest.fixture(scope='session')
def jaxtpc_is_synthetic():
    """True when the JAXTPC fixture falls back to the synthesizer."""
    return 'JAXTPC_DATA_ROOT' not in os.environ


@pytest.fixture(scope='session')
def lucid_is_synthetic():
    """True when the LUCiD fixture falls back to the synthesizer."""
    return 'LUCID_DATA_ROOT' not in os.environ

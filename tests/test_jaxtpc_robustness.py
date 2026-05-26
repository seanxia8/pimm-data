"""Robustness tests for JAXTPC readers against real production traits:

- output written with a blosc codec (the production default) must decode
  through the readers — guards the ``import hdf5plugin`` registration.
- a production-skipped event leaves a gap (missing ``event_NNN`` with
  ``n_events`` unchanged); readers must index present events, not arange.
"""
import os

import h5py
import numpy as np
import pytest

from pimm_data.testing import make_jaxtpc_sample
from pimm_data.readers.jaxtpc_edep import JAXTPCEdepReader
from pimm_data.readers.jaxtpc_hits import JAXTPCHitsReader

hdf5plugin = pytest.importorskip('hdf5plugin')


def _reencode(src, dst, comp):
    """Copy an HDF5 file, re-encoding array datasets with ``comp`` kwargs."""
    with h5py.File(src, 'r') as fs, h5py.File(dst, 'w') as fd:
        for k, v in fs.attrs.items():
            fd.attrs[k] = v

        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                g = fd.require_group(name)
                for k, v in obj.attrs.items():
                    g.attrs[k] = v
            else:
                kw = comp if (obj.ndim >= 1 and obj.nbytes >= 64) else {}
                d = fd.create_dataset(name, data=obj[()], **kw)
                for k, v in obj.attrs.items():
                    d.attrs[k] = v

        fs.visititems(visit)


def test_reader_reads_blosc_compressed(tmp_path):
    """Readers decode blosc-zstd output identically to uncompressed."""
    root = make_jaxtpc_sample(str(tmp_path / 'u'), n_events=2)
    base = JAXTPCEdepReader(data_root=os.path.join(root, 'edep'),
                            split='', dataset_name='sim').read_event(0)

    blosc = dict(hdf5plugin.Blosc(cname='zstd', clevel=4,
                                  shuffle=hdf5plugin.Blosc.SHUFFLE))
    bdir = tmp_path / 'b' / 'edep'
    bdir.mkdir(parents=True)
    _reencode(os.path.join(root, 'edep', 'sim_edep_0000.h5'),
              str(bdir / 'sim_edep_0000.h5'), blosc)

    got = JAXTPCEdepReader(data_root=str(bdir), split='',
                           dataset_name='sim').read_event(0)
    np.testing.assert_allclose(got['coord'], base['coord'])
    np.testing.assert_allclose(got['energy'], base['energy'])


@pytest.mark.parametrize('modality,reader_cls', [
    ('edep', JAXTPCEdepReader),
    ('hits', JAXTPCHitsReader),
])
def test_reader_tolerates_missing_event(tmp_path, modality, reader_cls):
    """A missing event_NNN (gap) is skipped, not arange'd into a KeyError."""
    root = make_jaxtpc_sample(str(tmp_path), n_events=3)
    path = os.path.join(root, modality, f'sim_{modality}_0000.h5')
    with h5py.File(path, 'r+') as f:
        assert int(f['config'].attrs['n_events']) == 3
        del f['event_001']  # production-style gap; n_events stays 3

    r = reader_cls(data_root=os.path.join(root, modality),
                   split='', dataset_name='sim')
    assert len(r) == 2                       # only present events counted
    assert r.indices[0].tolist() == [0, 2]   # gap skipped, order preserved
    r.read_event(0)                          # event_000
    r.read_event(1)                          # maps to event_002, no KeyError

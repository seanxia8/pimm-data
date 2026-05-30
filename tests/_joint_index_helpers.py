"""Shared assertions for the JAXTPC + LUCiD joint-index (Phase A) tests.

Both detectors have the same desync shape — one global ``idx`` fanned out to
each modality reader's own present-event index — and the same fix (intersect
present events across modalities, inject one shared index). The alignment check
is identical, so it lives here once instead of in each test file.
"""


def readers(ds):
    """The dataset's non-None modality readers (order-independent for the
    alignment check, which compares every reader to the first)."""
    return [r for r in (ds.edep_reader, ds.sensor_reader,
                        ds.hits_reader, ds.labl_reader) if r is not None]


def event_key_per_idx(ds):
    """Per global idx, the set of event_keys the modalities resolve to.

    ``_locate_event`` returns ``(f, event_key)`` or ``(f, event_key, n_volumes)``
    — ``event_key`` is element 1 in both. Alignment ⇒ every set has size 1.
    """
    rs = readers(ds)
    for r in rs:
        if not r._initted:
            r.h5py_worker_init()
    return [{r._locate_event(idx)[1] for r in rs} for idx in range(len(ds))]


def assert_aligned(ds):
    """All readers share the identical joint index and resolve idx→same event."""
    rs = readers(ds)
    ref_idx = [a.tolist() for a in rs[0].indices]
    ref_cum = rs[0].cumulative_lengths.tolist()
    for r in rs[1:]:
        assert [a.tolist() for a in r.indices] == ref_idx
        assert r.cumulative_lengths.tolist() == ref_cum
    for keys in event_key_per_idx(ds):
        assert len(keys) == 1, f"modalities disagree on the event: {keys}"

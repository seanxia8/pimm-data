"""Joint cross-modality event index (Phase A / D42).

Each modality reader indexes the ``event_*`` groups present in its own
shards and maps a global ``idx`` through *that* reader's index. A multimodal
dataset that passes one ``idx`` to every reader (with ``_n_events =
min(len(r))``) is only correct if every reader holds the SAME ordered list of
physics events. It does not, whenever the present-event sets diverge:

* an step event filter (``min_deposits`` / ``min_segments``) masks the step
  reader's index but not the others;
* a production gap (a skipped ``event_NNN``) is present in some modalities but
  not others.

Either way a global ``idx`` then resolves to *different physics events* in
different modalities — silently — corrupting every cross-modality join
(``deposit_to_track`` / ``group_to_track`` / ``bridges``).

:func:`build_joint_index` fixes this for any multimodal dataset: it intersects
the present event numbers across every loaded modality (per shard) and
overwrites each reader's ``indices`` / ``cumulative_lengths`` with the shared
joint index, so all readers resolve a global ``idx`` to the same
``(file, event_num)``. Shards align by sorted-glob position.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def build_joint_index(named_readers, *, strict_lengths=False,
                      source_label='', filter_label=''):
    """Intersect present events across modalities; inject one shared index.

    Parameters
    ----------
    named_readers : list[tuple[str, reader]]
        ``(modality_name, reader)`` for every loaded modality. Each reader
        must expose ``h5_files``, ``indices`` (list of per-shard ``np.int64``
        event-number arrays) and ``cumulative_lengths``.
    strict_lengths : bool
        If True, raise on any cross-modality shard-count or event mismatch
        instead of warning and aligning on the intersection.
    source_label : str
        Identifier for log/error messages (e.g. the dataset ``data_root``).
    filter_label : str
        Human description of any active event filter (e.g.
        ``"min_deposits=5"``) for the mismatch message.

    Returns
    -------
    int
        Total number of jointly-present events (the dataset's ``_n_events``).

    Side effects
    ------------
    Overwrites ``reader.indices``, ``reader.cumulative_lengths`` and trims
    ``reader.h5_files`` on every reader so they share the joint index.
    """
    readers = [r for _, r in named_readers]
    if not readers:
        return 0

    # Shards align by position in each modality's sorted file list (A4).
    shard_counts = {n: len(r.h5_files) for n, r in named_readers}
    n_files = min(shard_counts.values())
    if len(set(shard_counts.values())) > 1:
        msg = (f"{source_label}: shard-count mismatch across modalities "
               f"{shard_counts}; aligning on the first {n_files} shard(s).")
        if strict_lengths:
            raise ValueError(msg)
        log.warning(msg)

    raw_totals = {n: int(sum(len(r.indices[s]) for s in range(n_files)))
                  for n, r in named_readers}

    joint = []
    for s in range(n_files):
        common = {int(e) for e in readers[0].indices[s]}
        for r in readers[1:]:
            common &= {int(e) for e in r.indices[s]}
        joint.append(np.array(sorted(common), dtype=np.int64))

    cum = (np.cumsum([len(a) for a in joint]).astype(np.int64)
           if joint else np.zeros(0, dtype=np.int64))
    total = int(cum[-1]) if len(cum) else 0

    # A4: surface any event dropped to keep modalities aligned. Expected under
    # an step event filter; otherwise it flags a real cross-modality gap.
    if any(t != total for t in raw_totals.values()):
        extra = f" (or filtered by {filter_label})" if filter_label else ""
        msg = (f"{source_label}: joint cross-modality index = {total} events; "
               f"per-modality present counts {raw_totals}. Events not present "
               f"in every loaded modality{extra} are excluded to keep all "
               f"modalities aligned.")
        if strict_lengths:
            raise ValueError(msg)
        log.warning(msg)

    for r in readers:
        r.indices = joint
        r.cumulative_lengths = cum
        r.h5_files = r.h5_files[:n_files]
    return total

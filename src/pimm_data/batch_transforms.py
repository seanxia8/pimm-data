"""
Post-collate, on-device batch transforms — the runner pimm-data owns.

Workers stay CPU/sparse; the dense path runs here, once per batch, in the main
process. Post-collate transforms are built from config by
:func:`build_batch_transforms` (the batch-side analog of
:class:`pimm_data.transform.Compose`) and run by :func:`apply_batch_transforms` on
the collated tensors; the dense grids are born on-device. Device-agnostic: a
``ToDevice('cpu')`` stage (or ``device='cpu'``) runs the exact same transforms on
CPU (tests / no-GPU). This module imports ``torch`` but never ``torch.cuda``.

Built-in batch transforms (registered, ``scope='batch'``):

* ``ToDevice``         — move the batch to a device (device-as-transform)
* ``BatchDensify``     — sparse ``wire/time/value/plane_gid/offset`` -> ``batch[dense_key]={gid:(B,W,T)}``
* ``BatchAddIntrinsicNoise`` — fresh coherent (+optional incoherent) noise per event
* ``BatchDigitize``    — quantize to ADC codes

The dense sensor chain is just one config — :func:`sensor_dense_cfg` returns it as
``dict(type=…)`` entries to splat into :func:`build_batch_transforms`; there is no
bespoke builder it requires. pimm-data stages take ``(batch, *, seeds)``; a user
stage is a bare ``fn(batch) -> batch``. They mutate/return the batch dict:

* ``BatchDensify``      — sparse ``wire/time/value/plane_gid/offset`` -> ``batch[dense_key]={gid:(B,W,T)}``
* ``BatchAddIntrinsicNoise`` — fresh coherent (+optional incoherent) noise per event
* ``BatchDigitize``     — quantize to ADC codes

**User stages (the post-collate half of the uniform-transform path).** A stage is
just a callable. pimm-data's own stages take ``(batch, *, seeds)``; a *user* stage
is a bare ``fn(batch) -> batch`` (a lambda, an ``nn.Module``, a pre-built object) —
the runner inspects the signature (:func:`_accepts_seeds`) and omits ``seeds`` when
the callable doesn't declare it, so a GPU op is first-class with no registration and
no ``seeds`` plumbing, exactly like a pre-collate ``Compose`` entry. Contract for a
user stage: operate on the **collated, on-device** batch (tensors, ``offset``-keyed
ragged layout); keep any per-point array length-consistent with ``offset``; do not
drop ``name``/``split`` (the seed-identity carriers); a stage that needs the collated
set is ``scope='batch'`` (the default for these) and must NOT be placed in ``Compose``
(the build-time fence rejects it). Anything stateful/learned belongs to the model,
not here — the runner is an optional convenience, not a required phase.

Seeds are **content-addressed** per event (``blake2b(name) ^ base_seed ^ epoch ^
rank``), so the same event gets the same noise regardless of **batch position,
worker, or resume**. Caveats: ``rank`` is folded in to decorrelate DDP replicas,
so changing ``world_size`` (which re-partitions events across ranks) changes the
realization; and the **incoherent** component draws on a torch ``Generator`` whose
CPU and CUDA streams differ for the same seed, so incoherent realizations are
**device-specific** (coherent uses a numpy Generator and is device-independent /
bit-exact to JAXTPC). Geometry comes from a registry (e.g.
``JAXTPCDataset.plane_geometry()``) passed to the stages — not carried per-sample
through collate.
"""

from __future__ import annotations

import hashlib
import inspect
import warnings

import torch

from . import dense_ops
from .transform import TRANSFORMS
from .noise import (DEFAULT_ENC, DEFAULT_SAMPLING_RATE_HZ, DEFAULT_GROUP_SIZE,
                    DEFAULT_COH_RMS_ADC, DEFAULT_COH_CORNER_FREQ_HZ,
                    DEFAULT_COH_SLOPE, DEFAULT_COH_BETA)


# ── device move (idempotent) ──────────────────────────────────────────────
def move_to_device(batch, device, non_blocking=True):
    """Recursively move tensors to ``device``; non-tensors untouched.

    Idempotent — a tensor already on ``device`` is returned by ``.to`` without a
    copy. Use ``DataLoader(pin_memory=True)`` for ``non_blocking`` to be truly async.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device, non_blocking) for v in batch)
    return batch


# ── content-addressed seeding ─────────────────────────────────────────────
def content_seed(event_name, base_seed=0, epoch=0, rank=0):
    """Stable per-event seed: same event -> same noise across batch/worker/resume."""
    payload = f"{event_name}|{int(base_seed)}|{int(epoch)}|{int(rank)}".encode()
    h = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(h, 'little') & ((1 << 63) - 1)


def _batch_seeds(batch, base_seed, epoch, rank, n):
    names = batch.get('name')
    if isinstance(names, str):
        names = [names]
    if not names:
        # No per-event id -> we cannot content-address. A position-based fallback
        # is NOT reproducible across shuffles/runs, so warn loudly rather than
        # silently give wrong reproducibility.
        warnings.warn(
            "AddNoise: batch has no 'name' -> seeding falls back to batch "
            "position, which is NOT reproducible across shuffles/epochs/runs. "
            "Ensure Collect passes 'name' through.", RuntimeWarning, stacklevel=2)
        names = [f"_idx{i}" for i in range(n)]
    return [content_seed(nm, base_seed, epoch, rank) for nm in names]


def _batch_size(batch, offset_key='offset'):
    off = batch.get(offset_key)
    if off is not None:
        return int(off.numel())
    return 1


def _accepts_seeds(stage):
    """Does this post-collate stage take a ``seeds=`` kwarg?

    pimm-data's own stages (``BatchDensify`` …) take ``(batch, *, seeds)``; a
    *user* stage is just a bare callable ``fn(batch) -> batch`` (a lambda, an
    ``nn.Module``, a pre-built transform), exactly like a pre-collate ``Compose``
    entry. The runner inspects the signature so both forms are first-class on the
    post-collate side — no registration, no required ``seeds`` plumbing. A stage
    with ``**kwargs`` is assumed to absorb ``seeds`` (our protocol); an
    un-introspectable callable (C/builtin) is assumed to follow the stage protocol.
    """
    fn = stage if (inspect.isfunction(stage) or inspect.ismethod(stage)) \
        else getattr(stage, '__call__', stage)
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return True
    if 'seeds' in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# ── batch transforms (post-collate; registered, scope='batch') ───────────────
@TRANSFORMS.register_module()
class ToDevice:
    """Move the whole batch to a device — the device move expressed as a transform.

    So device selection composes uniformly with the other batch transforms instead
    of being a special ``apply_batch_transforms(device=)`` argument: put
    ``dict(type='ToDevice', device='cuda')`` first in the list and everything after
    runs on-device. Idempotent (a tensor already on ``device`` is returned without a
    copy). ``scope='batch'`` so the pre-collate ``Compose`` fence keeps it post-collate.
    """

    scope = 'batch'

    def __init__(self, device, *, non_blocking=True):
        self.device = device
        self.non_blocking = non_blocking

    def __call__(self, batch):
        return move_to_device(batch, self.device, non_blocking=self.non_blocking)


@TRANSFORMS.register_module()
class BatchDensify:
    """Sparse hits -> per-plane dense grids ``batch[dense_key] = {gid: (B,W,T)}``."""

    scope = 'batch'  # needs the collated set (B, offset) — post-collate only

    def __init__(self, geom, *, modality=None, wire_key='wire', time_key='time',
                 value_key='value', plane_key='plane_gid', offset_key='offset',
                 dense_key='sensor_dense'):
        self.geom = geom
        self.modality = modality
        self.wire_key, self.time_key, self.value_key = wire_key, time_key, value_key
        self.plane_key, self.offset_key, self.dense_key = plane_key, offset_key, dense_key

    def __call__(self, batch, *, seeds=None):
        # modality=None → operate on the bare batch (back-compat); modality='X'
        # → scope to the namespaced batch['X'] sub-dict (multi-modality path).
        tgt = batch[self.modality] if self.modality is not None else batch
        tgt[self.dense_key] = dense_ops.densify(
            tgt[self.wire_key], tgt[self.time_key], tgt[self.value_key],
            tgt[self.plane_key], tgt[self.offset_key], self.geom)
        return batch


@TRANSFORMS.register_module()
class BatchAddIntrinsicNoise:
    """Add fresh per-event coherent (+optional incoherent) noise to the grids."""

    scope = 'batch'  # operates on the batched dense grid

    def __init__(self, geom, *, modality=None, coherent=True, incoherent=False,
                 dense_key='sensor_dense', enc=DEFAULT_ENC,
                 sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ,
                 group_size=DEFAULT_GROUP_SIZE, coh_rms=DEFAULT_COH_RMS_ADC,
                 coh_corner_freq_hz=DEFAULT_COH_CORNER_FREQ_HZ,
                 coh_spectral_slope=DEFAULT_COH_SLOPE, beta=DEFAULT_COH_BETA,
                 series_spectrum=None):
        self.geom = geom
        self.modality = modality
        self.coherent, self.incoherent = bool(coherent), bool(incoherent)
        self.dense_key = dense_key
        self.enc = tuple(enc)
        self.kw = dict(sampling_rate_hz=sampling_rate_hz, group_size=group_size,
                       coh_rms=coh_rms, coh_corner_freq_hz=coh_corner_freq_hz,
                       coh_spectral_slope=coh_spectral_slope, beta=beta,
                       series_spectrum=series_spectrum)

    def __call__(self, batch, *, seeds):
        tgt = batch[self.modality] if self.modality is not None else batch
        dense_ops.add_intrinsic_noise(
            tgt[self.dense_key], self.geom, seeds=seeds, enc=self.enc,
            coherent=self.coherent, incoherent=self.incoherent, **self.kw)
        return batch


@TRANSFORMS.register_module()
class BatchDigitize:
    """Quantize per-plane grids to ADC codes (pedestal from registry or override)."""

    scope = 'batch'  # operates on the batched dense grid

    def __init__(self, geom=None, *, modality=None, pedestal=None, n_bits=12,
                 adc_max=None, gain=1.0, dense_key='sensor_dense'):
        self.geom = geom or {}
        self.modality = modality
        self.pedestal = pedestal
        self.n_bits, self.adc_max, self.gain = n_bits, adc_max, gain
        self.dense_key = dense_key

    def __call__(self, batch, *, seeds=None):
        tgt = batch[self.modality] if self.modality is not None else batch
        ped = self.pedestal
        if ped is None:
            ped = {gid: e.get('pedestal', 0) for gid, e in self.geom.items()}
        tgt[self.dense_key] = dense_ops.digitize(
            tgt[self.dense_key], ped, n_bits=self.n_bits,
            adc_max=self.adc_max, gain=self.gain)
        return batch


# ── builder (the post-collate analog of transform.Compose) ───────────────────
def build_batch_transforms(cfg):
    """Build a post-collate transform list from config — the batch-side analog of
    :class:`pimm_data.transform.Compose`.

    Each entry is either a registry config ``dict(type='BatchDensify', …)`` (any
    ``scope='batch'`` transform — ``ToDevice``, ``BatchDensify``,
    ``BatchAddIntrinsicNoise``, ``BatchDigitize``, or your own registered one) or a
    bare callable (a user GPU op / ``nn.Module``). Returns the list to hand to
    :func:`apply_batch_transforms`. This is the general mechanism; the dense sensor
    chain is just one config you can build with it (see :func:`sensor_dense_cfg`).

    Unlike ``Compose`` (pre-collate), this is where a ``scope='batch'`` transform
    belongs; a ``scope='sample'`` entry is built but **warned** (it almost always
    wants the pre-collate ``Compose`` instead).
    """
    stages = []
    for entry in (cfg or []):
        if isinstance(entry, dict):
            t = TRANSFORMS.build(entry)
        elif callable(entry):
            t = entry
        else:
            raise TypeError(
                "build_batch_transforms entries must be dict (registry config) or "
                f"callable, got {type(entry).__name__}")
        if getattr(t, 'scope', 'batch') == 'sample':
            warnings.warn(
                f"{type(t).__name__} is scope='sample' (per-event) — it usually "
                "belongs in the pre-collate Compose, not a batch transform list.",
                RuntimeWarning, stacklevel=2)
        stages.append(t)
    return stages


def sensor_dense_cfg(geom, *, modality='sensor', device=None, coherent=True,
                     incoherent=False, digitize=True, n_bits=12, dense_key=None,
                     **noise_kw):
    """Optional sugar: the **config list** for the standard dense sensor chain
    ``[ToDevice?, BatchDensify, BatchAddIntrinsicNoise, BatchDigitize?]``.

    Returns plain ``dict(type=…)`` configs so you can splat it into
    :func:`build_batch_transforms` and add your own transforms around it::

        stages = build_batch_transforms([
            *sensor_dense_cfg(geom, device='cuda'),
            my_user_gpu_fn,
        ])

    ``geom`` is a canonical-plane-id registry (``JAXTPCDataset.plane_geometry()`` or
    ``load_plane_registry(json)``). Pass ``device=`` to prepend a ``ToDevice`` stage
    (the device-as-transform path) instead of the ``apply_batch_transforms(device=)``
    argument. ``modality='sensor'`` writes ``batch['sensor']['dense']``;
    ``modality=None`` operates on the bare batch (writes ``batch['sensor_dense']``).
    """
    if dense_key is None:
        dense_key = 'dense' if modality is not None else 'sensor_dense'
    cfg = []
    if device is not None:
        cfg.append(dict(type='ToDevice', device=device))
    cfg.append(dict(type='BatchDensify', geom=geom, modality=modality,
                    dense_key=dense_key))
    cfg.append(dict(type='BatchAddIntrinsicNoise', geom=geom, modality=modality,
                    coherent=coherent, incoherent=incoherent, dense_key=dense_key,
                    **noise_kw))
    if digitize:
        cfg.append(dict(type='BatchDigitize', geom=geom, modality=modality,
                        n_bits=n_bits, dense_key=dense_key))
    return cfg


def build_sensor_gpu_stages(geom, *, modality=None, **kw):
    """Back-compat thin wrapper: ``build_batch_transforms(sensor_dense_cfg(geom, …))``.

    Keeps the original ``modality=None`` (bare-batch) default — note
    :func:`sensor_dense_cfg` itself defaults to the namespaced ``modality='sensor'``.
    Prefer composing your own list with :func:`build_batch_transforms` +
    :func:`sensor_dense_cfg`; this is kept so existing callers keep working.
    """
    return build_batch_transforms(sensor_dense_cfg(geom, modality=modality, **kw))


# ── runner ──────────────────────────────────────────────────────────────────
def apply_batch_transforms(batch, stages, *, device=None, base_seed=0, epoch=0,
                           rank=0, offset_key='offset'):
    """Run ``stages`` (a list from :func:`build_batch_transforms`) on the batch.

    Per-event seeds are derived from ``batch['name']`` + ``base_seed/epoch/rank`` and
    passed to stages that declare ``seeds=``; a bare user callable ``fn(batch)`` is
    called without it.

    Device: the **preferred** way is a ``ToDevice`` transform as the first stage
    (device-as-transform — uniform with the rest). The ``device=`` argument is a
    back-compat convenience that prepends the move; if both are given the argument
    moves first and a leading ``ToDevice`` is then a no-op. ``stages`` empty +
    ``device`` set -> just the move.
    """
    if device is not None:
        batch = move_to_device(batch, device)
    if not stages:
        return batch
    seeds = _batch_seeds(batch, base_seed, epoch, rank, _batch_size(batch, offset_key))
    for stage in stages:
        # pimm-data stages take seeds=; a user's bare callable fn(batch) does not.
        # Both are first-class on the post-collate side (see _accepts_seeds).
        batch = stage(batch, seeds=seeds) if _accepts_seeds(stage) else stage(batch)
    return batch


class BatchTransformMixin:
    """Mixin for a LightningModule: run the stages in ``on_after_batch_transfer``.

    Set ``self.batch_stages`` (from :func:`build_sensor_gpu_stages`) and
    ``self.base_seed``. Lightning calls this with the batch already on-device, so
    ``device=None`` (no move). For a custom (non-Lightning) loop, call
    :func:`apply_batch_transforms` inline after the ``.to(device)`` instead.
    """

    batch_stages: list = []
    base_seed: int = 0

    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        trainer = getattr(self, 'trainer', None)
        epoch = getattr(trainer, 'current_epoch', 0) or 0
        rank = getattr(trainer, 'global_rank', 0) or 0
        return apply_batch_transforms(
            batch, self.batch_stages, device=None,
            base_seed=self.base_seed, epoch=epoch, rank=rank)

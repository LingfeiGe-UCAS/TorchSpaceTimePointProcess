"""Short-lived evaluation results shared by inference algorithms."""
from dataclasses import dataclass, field
from typing import Any

import torch


def _signature(model, catalog, state):
    def token(value):
        if isinstance(value, torch.Tensor):
            return (id(value), value._version, tuple(value.shape), value.dtype, value.device)
        return value
    tensors = []
    for owner, names in (
        (catalog, ('time', 'position', 'magnitude', 'depth', 'completeness_magnitude', 'finite_source_id')),
        (state, ('position', 'bandwidth', 'weights', 'depth', 'duration')),
        (model, ('patch_tau', 'patch_tau_total')),
    ):
        tensors.extend(token(getattr(owner, name, None)) for name in names)
    for method in ('parameters', 'buffers'):
        getter = getattr(model, method, None)
        if callable(getter):
            tensors.extend(token(value) for value in getter())
    return tuple(tensors)


@dataclass(frozen=True)
class EventIntensity:
    """Total intensity for one model/catalog/background snapshot.

    Values follow catalog ROW order, not event IDs. Uncomputed rows on a
    distributed rank are NaN. This is not a persistent model/background state:
    consume it during the same E step, before updating parameters or tau.
    Tensor versions guard common accidental reuse after in-place updates.
    """
    values: torch.Tensor
    model: Any = field(repr=False, compare=False)
    catalog: Any = field(repr=False, compare=False)
    background_state: Any = field(repr=False, compare=False)
    _snapshot: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if (self.values.shape != self.catalog.time.shape
                or self.values.device != self.catalog.time.device
                or self.values.dtype != self.catalog.time.dtype):
            raise ValueError('event intensity must match catalog rows, dtype and device')
        if self.values.requires_grad:
            raise ValueError('E-step intensity must be detached')
        object.__setattr__(self, '_snapshot', _signature(self.model, self.catalog, self.background_state))

    def validate_for(self, model, catalog, state, ranges):
        if model is not self.model or catalog is not self.catalog or state is not self.background_state:
            raise ValueError('event intensity belongs to another E-step snapshot')
        if self._snapshot != _signature(model, catalog, state):
            raise ValueError('event intensity is stale after model/catalog/state mutation')
        for begin, end in ranges:
            values = self.values[begin:end]
            if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
                raise ValueError('event intensity does not cover the requested target rows')

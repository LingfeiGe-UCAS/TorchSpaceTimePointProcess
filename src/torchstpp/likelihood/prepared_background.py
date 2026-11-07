"""M-step-only reuse of fixed KDE values, independent of trainable mu.

Prepared views are local to one M step. They neither mutate the catalog/state
nor install a persistent cache on a background/model/likelihood instance.
"""
from __future__ import annotations

import torch

from ..models.backgrounds import GaussianKDEBackground, GaussianBetaKDEBackground


class _CatalogView:
    def __init__(self, catalog, values):
        self._catalog = catalog
        self._m_step_background_values = values

    def __getattr__(self, name):
        return getattr(self._catalog, name)

    def __len__(self):
        return len(self._catalog.time)

    def take(self, index):
        return _CatalogView(self._catalog.take(index), self._m_step_background_values[index])


class _StateView:
    def __init__(self, state, integral):
        self._state = state
        self._m_step_background_integral = integral

    def __getattr__(self, name):
        return getattr(self._state, name)


class _PreparedMstep:
    def __init__(self, likelihood, model, catalog, domain, state, values, integral):
        self.likelihood = likelihood
        self.originals = (model, catalog, domain, state, model.background)
        self.catalog = _CatalogView(catalog, values)
        self.state = _StateView(state, integral)

    def __call__(self, model, catalog, domain, state):
        if any(a is not b for a, b in zip(
            (model, catalog, domain, state, model.background), self.originals
        )):
            raise ValueError("prepared M-step background belongs to another fitting state")
        return self.likelihood(model, self.catalog, domain, self.state)


def prepare_m_step(likelihood, model, catalog, domain, state):
    """Prepare fixed built-in KDE quantities for this M step only.

    Custom backgrounds are left unchanged: their parameters may be trainable.
    Built-in KDE inputs or integrators requiring gradients also disable reuse.
    Distributed preparation follows the same fitting-target schedule as local
    likelihood evaluation, including pair budgets and empty ranks.
    """
    if type(model.background) not in (GaussianKDEBackground, GaussianBetaKDEBackground):
        return likelihood
    exact = getattr(likelihood, "exact", likelihood)
    exact._validate_inputs(catalog, domain)
    target_indices = torch.nonzero(domain.target_mask(catalog), as_tuple=False).flatten()
    targets = catalog.take(target_indices)
    sources = exact.select_sources(catalog, domain)
    schedule = getattr(likelihood, "local_target_schedule", None)
    if callable(schedule):
        ranges, target_chunk, _ = schedule(
            targets.time, sources.time, background_source_count=int(state.position.shape[0])
        )
    else:
        ranges = ((0, len(targets.time)),)
        target_chunk = exact.target_chunk_size
    # NaNs make any accidental access to an unprepared/nonlocal row fail loudly.
    # Values follow catalog row indices, never potentially duplicated event IDs.
    values = torch.full_like(catalog.time, float("nan"))
    with torch.enable_grad():
        for begin, end in ranges:
            for start in range(begin, end, target_chunk):
                stop = min(start + target_chunk, end)
                block = model.background.evaluate(targets.take(slice(start, stop)), state)
                if block.requires_grad:
                    return likelihood
                values[target_indices[start:stop]] = block.detach()
        integral = model.background.integrate(state, domain)
        if integral.requires_grad:
            return likelihood
    return _PreparedMstep(likelihood, model, catalog, domain, state, values, integral.detach())

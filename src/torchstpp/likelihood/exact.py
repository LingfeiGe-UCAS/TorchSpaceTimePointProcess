from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .terms import LikelihoodTerms


class ExactLikelihood(nn.Module):
    """Exact single-device point-process likelihood with bounded-memory blocks."""

    def __init__(
        self,
        *,
        target_chunk_size: int = 2048,
        source_chunk_size: int = 4096,
        checkpoint_blocks: bool = False,
    ) -> None:
        super().__init__()
        if target_chunk_size < 1 or source_chunk_size < 1:
            raise ValueError("chunk sizes must be positive")
        self.target_chunk_size = target_chunk_size
        self.source_chunk_size = source_chunk_size
        self.checkpoint_blocks = checkpoint_blocks

    def prepare_m_step(self, model, catalog, domain, background_state):
        """Reuse fixed background density/integral within one optimizer step."""
        from .prepared_background import prepare_m_step
        return prepare_m_step(self, model, catalog, domain, background_state)

    def _run_block(self, model: Any, function) -> torch.Tensor:
        """Optionally recompute a block during backward to reduce memory."""
        # Models opting into fine-grained recomputation own both their point
        # and patch branches. Do not nest a coarse event checkpoint around them.
        if getattr(model, "checkpoint_patches", False):
            return function()
        if not self.checkpoint_blocks or not torch.is_grad_enabled():
            return function()

        anchor = next(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            None,
        )
        if anchor is None:
            return function()

        # The non-reentrant implementation supports parameters captured by the
        # closure. The zero anchor keeps the checkpoint connected explicitly
        # while leaving the scientific value unchanged.
        return checkpoint(
            lambda parameter_anchor: function() + parameter_anchor.sum() * 0.0,
            anchor,
            use_reentrant=False,
        )

    @staticmethod
    def _validate_inputs(catalog: Any, domain: Any) -> None:
        catalog_required = (
            "time", "position", "magnitude", "completeness_magnitude", "take"
        )
        missing = [name for name in catalog_required if not hasattr(catalog, name)]
        if missing:
            raise TypeError(f"catalog is missing required attributes: {missing}")
        domain_required = (
            "time_zero",
            "time_start",
            "time_end",
            "target_mask",
            "source_mask",
        )
        missing = [name for name in domain_required if not hasattr(domain, name)]
        if missing:
            raise TypeError(f"domain is missing required attributes: {missing}")
        if catalog.time.ndim != 1:
            raise ValueError("catalog.time must have shape [N]")
        if catalog.position.shape != (catalog.time.shape[0], 2):
            raise ValueError(
                "catalog.position must contain horizontal coordinates with shape [N, 2]"
            )
        if catalog.magnitude.shape != catalog.time.shape:
            raise ValueError("catalog.magnitude must have shape [N]")
        if torch.any(catalog.magnitude < catalog.completeness_magnitude):
            raise ValueError(
                "catalog contains events below completeness_magnitude; call "
                "catalog.filter_for_analysis(...) before likelihood evaluation"
            )
        depth = getattr(catalog, "depth", None)
        if depth is not None and depth.shape != catalog.time.shape:
            raise ValueError("catalog.depth must be None or have shape [N]")
        if catalog.time.numel() > 1 and torch.any(catalog.time[1:] < catalog.time[:-1]):
            raise ValueError("catalog must be sorted by non-decreasing time")
        if domain.time_start < domain.time_zero:
            raise ValueError("domain.time_start must not be earlier than time_zero")
        if domain.time_end <= domain.time_start:
            raise ValueError("domain.time_end must be later than time_start")
        if torch.any(catalog.time < domain.time_zero) or torch.any(
            catalog.time > domain.time_end
        ):
            raise ValueError(
                "catalog contains events outside [domain.time_zero, "
                "domain.time_end]; call catalog.filter_for_analysis(...)"
            )

    def select_targets(self, catalog: Any, domain: Any) -> Any:
        # Prepared M-step views preserve row indexing; domain validation still
        # receives the original catalog type (including strict EventCatalogs).
        mask = domain.target_mask(getattr(catalog, "_catalog", catalog))
        if mask.shape != catalog.time.shape or mask.dtype != torch.bool:
            raise ValueError("domain.target_mask(catalog) must return bool Tensor [N]")
        if not bool(torch.any(mask)):
            raise ValueError("the observation domain contains no target events")
        return catalog.take(mask)

    def select_sources(self, catalog: Any, domain: Any) -> Any:
        mask = domain.source_mask(getattr(catalog, "_catalog", catalog))
        if mask.shape != catalog.time.shape or mask.dtype != torch.bool:
            raise ValueError("domain.source_mask(catalog) must return bool Tensor [N]")
        if not bool(torch.any(mask)):
            raise ValueError("the source-history window contains no events")
        # Source kernels/integrators may inspect concrete catalog fields. Only
        # targets need cached background values; sources retain their real type.
        return getattr(catalog, "_catalog", catalog).take(mask)

    @staticmethod
    def _block(data: Any, begin: int, end: int) -> Any:
        return data.take(slice(begin, end))

    def event_log_intensity(
        self,
        model: Any,
        targets: Any,
        sources: Any,
        background_state: Any,
    ) -> torch.Tensor:
        """Compute the exact event term, distributing work by target blocks."""
        total = targets.time.new_zeros(())
        target_count = targets.time.shape[0]
        source_count = sources.time.shape[0]

        for target_begin in range(0, target_count, self.target_chunk_size):
            target_end = min(target_begin + self.target_chunk_size, target_count)
            target_block = self._block(targets, target_begin, target_end)
            background_values = getattr(target_block, "_m_step_background_values", None)
            if background_values is None:
                background_values = model.background.evaluate(target_block, background_state)
            intensity = model.background_rate(background_values)

            for source_begin in range(0, source_count, self.source_chunk_size):
                source_end = min(source_begin + self.source_chunk_size, source_count)
                source_block = self._block(sources, source_begin, source_end)

                # Sorted times let us skip this and every later source block.
                if source_block.time[0] > target_block.time[-1]:
                    break
                pairwise_sum = self._run_block(
                    model,
                    lambda saved_target=target_block, saved_source=source_block:
                        model.pairwise_rate(saved_target, saved_source).sum(dim=-1),
                )
                intensity = intensity + pairwise_sum

            if not torch.all(torch.isfinite(intensity)):
                raise FloatingPointError("event intensity contains non-finite values")
            if torch.any(intensity <= 0):
                raise FloatingPointError("event intensity must be strictly positive")
            total = total + torch.log(intensity).sum()
        return total

    def compensator(
        self,
        model: Any,
        sources: Any,
        domain: Any,
        background_state: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        background_term = self.background_compensator(
            model, domain, background_state
        )
        triggering_term = self.triggering_compensator(model, sources, domain)
        return background_term, triggering_term

    def background_compensator(
        self,
        model: Any,
        domain: Any,
        background_state: Any,
    ) -> torch.Tensor:
        """Return the complete background compensator."""

        background_mass = getattr(background_state, "_m_step_background_integral", None)
        if background_mass is None:
            background_mass = model.background.integrate(background_state, domain)
        return model.integrated_background(background_mass)

    def triggering_compensator(
        self,
        model: Any,
        sources: Any,
        domain: Any,
    ) -> torch.Tensor:
        """Return the triggering compensator for the supplied source events."""

        triggering_term = sources.time.new_zeros(())
        source_count = sources.time.shape[0]
        for source_begin in range(0, source_count, self.source_chunk_size):
            source_end = min(source_begin + self.source_chunk_size, source_count)
            source_block = self._block(sources, source_begin, source_end)
            source_integral = self._run_block(
                model,
                lambda saved_source=source_block:
                    model.integrated_source(saved_source, domain).sum(),
            )
            triggering_term = triggering_term + source_integral
        return triggering_term

    def terms(
        self,
        model: Any,
        catalog: Any,
        domain: Any,
        background_state: Any,
    ) -> LikelihoodTerms:
        self._validate_inputs(catalog, domain)
        targets = self.select_targets(catalog, domain)
        sources = self.select_sources(catalog, domain)
        if targets.time.numel() == 0:
            raise ValueError("the observation domain contains no target events")

        log_intensity = self.event_log_intensity(
            model, targets, sources, background_state
        )
        background_term, triggering_term = self.compensator(
            model, sources, domain, background_state
        )
        negative_log_likelihood = (
            background_term + triggering_term - log_intensity
        )
        return LikelihoodTerms(
            negative_log_likelihood=negative_log_likelihood,
            log_intensity=log_intensity,
            background_compensator=background_term,
            triggering_compensator=triggering_term,
            target_count=targets.time.shape[0],
        )

    def forward(
        self,
        model: Any,
        catalog: Any,
        domain: Any,
        background_state: Any,
    ) -> torch.Tensor:
        return self.terms(
            model, catalog, domain, background_state
        ).negative_log_likelihood

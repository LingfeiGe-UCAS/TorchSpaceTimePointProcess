"""Mixed point- and finite-source three-dimensional ETAS model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ...finite_source import FiniteSourceGeometry
from ..components import conditional_beta_depth_density
from ..components.fixed_inputs import FixedInputCache
from ..parameters import ModelParameter, ParameterConfig
from .etas3d import ETAS3D, ETAS3DParameters


@dataclass(frozen=True)
class FiniteSourceETAS3DParameters(ETAS3DParameters):
    """ETAS3D parameters plus the two legacy finite-source kernel scales."""

    chara_d: ParameterConfig
    ch_eta: ParameterConfig

    @classmethod
    def defaults(cls) -> "FiniteSourceETAS3DParameters":
        base = ETAS3DParameters.defaults()
        return cls(
            mu=base.mu,
            amplitude=base.amplitude,
            alpha=base.alpha,
            c=base.c,
            p=base.p,
            d=base.d,
            q=base.q,
            gamma=base.gamma,
            eta=base.eta,
            delta=base.delta,
            chara_d=ParameterConfig(0.5, lower_bound=0.0),
            ch_eta=ParameterConfig(10.0, lower_bound=0.0),
        )


class FiniteSourceETAS3D(ETAS3D):
    """Fortran-compatible ETAS3D with a ragged finite-source patch bank.

    Catalog events whose ``finite_source_id`` is zero retain the point-source
    ETAS3D kernel. A positive identifier selects one contiguous patch group.
    Patch rates are weighted by ``patch_tau / patch_tau_total`` and reduced back
    to one column per physical source event, so the existing exact likelihood,
    result, and distributed interfaces remain event based.

    ``patch_tau`` is an EM state buffer rather than a trainable parameter. It is
    held fixed during every M step and can be updated in place through
    :meth:`set_patch_tau_`.
    """

    def __init__(
        self,
        parameters: FiniteSourceETAS3DParameters | None = None,
        *,
        finite_source_geometry: FiniteSourceGeometry,
        maximum_depth: float,
        background_depth_concentration: float | None = None,
        depth_boundary_offset: float = 0.5,
        depth_reference_magnitude_offset: float = 0.0,
        background: Any | None = None,
        spatial_integrator: Any | None = None,
        minimum_spatial_scale: float = 0.0,
        patch_chunk_size: int = 4096,
        checkpoint_patches: bool = False,
        horizontal_smoothing_bandwidth: float = 0.015,
        depth_smoothing_concentration: float = 100.0,
        tau_normalization: str = "fortran",
        smoothing_method: str = "blocked",
        zero_responsibility_policy: str = "retain",
        patch_tau: torch.Tensor | None = None,
        patch_smoothing_normalizer: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(finite_source_geometry, FiniteSourceGeometry):
            raise TypeError(
                "finite_source_geometry must be a FiniteSourceGeometry"
            )
        if patch_chunk_size < 1:
            raise ValueError("patch_chunk_size must be positive")
        self.checkpoint_patches = bool(checkpoint_patches)
        if (
            not math.isfinite(horizontal_smoothing_bandwidth)
            or horizontal_smoothing_bandwidth <= 0.0
        ):
            raise ValueError(
                "horizontal_smoothing_bandwidth must be finite and positive"
            )
        if (
            not math.isfinite(depth_smoothing_concentration)
            or depth_smoothing_concentration < 0.0
        ):
            raise ValueError(
                "depth_smoothing_concentration must be finite and non-negative"
            )
        tau_normalization = str(tau_normalization).lower()
        smoothing_method = str(smoothing_method).lower()
        if smoothing_method not in {"blocked", "fft"}:
            raise ValueError("smoothing_method must be 'blocked' or 'fft'")
        if tau_normalization not in {"fortran", "corrected"}:
            raise ValueError(
                "tau_normalization must be 'fortran' or 'corrected'"
            )
        if zero_responsibility_policy == "keep":
            zero_responsibility_policy = "retain"
        if zero_responsibility_policy not in {"retain", "uniform"}:
            raise ValueError(
                "zero_responsibility_policy must be 'retain' or 'uniform'"
            )
        if bool(torch.any(finite_source_geometry.depth > maximum_depth)):
            raise ValueError(
                "finite-source patch depth must lie in [0, maximum_depth]"
            )

        parameters = parameters or FiniteSourceETAS3DParameters.defaults()
        if not isinstance(parameters, FiniteSourceETAS3DParameters):
            raise TypeError(
                "parameters must be a FiniteSourceETAS3DParameters instance"
            )
        super().__init__(
            parameters,
            maximum_depth=maximum_depth,
            background_depth_concentration=background_depth_concentration,
            depth_boundary_offset=depth_boundary_offset,
            depth_reference_magnitude_offset=depth_reference_magnitude_offset,
            background=background,
            spatial_integrator=spatial_integrator,
            minimum_spatial_scale=minimum_spatial_scale,
        )

        self.chara_d_parameter = ModelParameter(
            parameters.chara_d, lower_bound=0.0
        )
        self.ch_eta_parameter = ModelParameter(
            parameters.ch_eta, lower_bound=0.0
        )
        self.patch_chunk_size = int(patch_chunk_size)
        self.horizontal_smoothing_bandwidth = float(
            horizontal_smoothing_bandwidth
        )
        self.depth_smoothing_concentration = float(
            depth_smoothing_concentration
        )
        self.tau_normalization = tau_normalization
        self.smoothing_method = smoothing_method
        self.zero_responsibility_policy = zero_responsibility_policy
        self.finite_source_metadata = dict(finite_source_geometry.metadata)

        # Register geometry with the model so ordinary Module.to(), state_dict(),
        # replica validation, and multi-GPU initialization include every patch.
        self.register_buffer(
            "patch_id", finite_source_geometry.patch_id.detach().clone()
        )
        self.register_buffer(
            "patch_finite_source_id",
            finite_source_geometry.finite_source_id.detach().clone(),
        )
        self.register_buffer(
            "patch_position", finite_source_geometry.position.detach().clone()
        )
        self.register_buffer(
            "patch_depth", finite_source_geometry.depth.detach().clone()
        )
        self.register_buffer(
            "finite_source_group_ids",
            finite_source_geometry.group_ids.detach().clone(),
        )
        self.register_buffer(
            "finite_source_group_offsets",
            finite_source_geometry.group_offsets.detach().clone(),
        )

        initial_tau = (
            self.patch_position.new_ones(len(finite_source_geometry))
            if patch_tau is None
            else torch.as_tensor(
                patch_tau,
                dtype=self.patch_position.dtype,
                device=self.patch_position.device,
            ).detach().clone()
        )
        self._validate_patch_vector(initial_tau, "patch_tau", nonnegative=True)
        initial_totals = self._group_totals(initial_tau)
        if bool(torch.any(initial_totals <= 0.0)):
            raise ValueError("initial patch_tau must have positive mass in every group")
        self.register_buffer("patch_tau", initial_tau)
        self.register_buffer("patch_tau_total", initial_totals)

        smoothing_normalizer = (
            self.patch_position.new_ones(len(finite_source_geometry))
            if patch_smoothing_normalizer is None
            else torch.as_tensor(
                patch_smoothing_normalizer,
                dtype=self.patch_position.dtype,
                device=self.patch_position.device,
            ).detach().clone()
        )
        self._validate_patch_vector(
            smoothing_normalizer,
            "patch_smoothing_normalizer",
            positive=True,
        )
        self.register_buffer(
            "patch_smoothing_normalizer", smoothing_normalizer
        )

        # Geometry IDs are immutable model structure, so a small Python lookup is
        # safe and avoids a CUDA synchronization for every source block.
        group_ids = finite_source_geometry.group_ids.detach().cpu().tolist()
        offsets = finite_source_geometry.group_offsets.detach().cpu().tolist()
        self._group_slices = {
            int(group_id): slice(int(offsets[index]), int(offsets[index + 1]))
            for index, group_id in enumerate(group_ids)
        }
        self._group_ordinals = {
            int(group_id): index for index, group_id in enumerate(group_ids)
        }

    @property
    def chara_d(self) -> torch.Tensor:
        """Finite-source horizontal scale (the legacy ``chara_d``)."""

        return self.chara_d_parameter()

    @property
    def ch_eta(self) -> torch.Tensor:
        """Finite-source conditional-Beta depth concentration."""

        return self.ch_eta_parameter()

    @property
    def patch_count(self) -> int:
        return int(self.patch_id.numel())

    @property
    def finite_source_group_count(self) -> int:
        return int(self.finite_source_group_ids.numel())

    @property
    def normalized_patch_tau(self) -> torch.Tensor:
        totals = torch.repeat_interleave(
            self.patch_tau_total,
            self.finite_source_group_offsets[1:]
            - self.finite_source_group_offsets[:-1],
        )
        return self.patch_tau / totals

    @property
    def finite_source_geometry(self) -> SimpleNamespace:
        """A lightweight live view of geometry buffers for diagnostics/E steps."""

        return SimpleNamespace(
            patch_id=self.patch_id,
            finite_source_id=self.patch_finite_source_id,
            position=self.patch_position,
            depth=self.patch_depth,
            group_ids=self.finite_source_group_ids,
            group_offsets=self.finite_source_group_offsets,
        )

    def parameter_values(self) -> dict[str, torch.Tensor]:
        values = super().parameter_values()
        values.update(chara_d=self.chara_d, ch_eta=self.ch_eta)
        return values

    def optimization_parameter_values(self) -> dict[str, torch.Tensor]:
        values = super().optimization_parameter_values()
        values.update(chara_d=self.chara_d, ch_eta=self.ch_eta)
        return values

    def _validate_patch_vector(
        self,
        value: torch.Tensor,
        name: str,
        *,
        nonnegative: bool = False,
        positive: bool = False,
    ) -> None:
        if value.shape != (self.patch_count,):
            raise ValueError(f"{name} must have shape [{self.patch_count}]")
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if not bool(torch.all(torch.isfinite(value))):
            raise ValueError(f"{name} must contain only finite values")
        if nonnegative and bool(torch.any(value < 0.0)):
            raise ValueError(f"{name} must be non-negative")
        if positive and bool(torch.any(value <= 0.0)):
            raise ValueError(f"{name} must be positive")

    def _group_totals(self, tau: torch.Tensor) -> torch.Tensor:
        totals = []
        offsets = self.finite_source_group_offsets.detach().cpu().tolist()
        for begin, end in zip(offsets[:-1], offsets[1:]):
            totals.append(tau[int(begin):int(end)].sum())
        return torch.stack(totals)

    @torch.no_grad()
    def set_patch_tau_(self, tau: torch.Tensor) -> "FiniteSourceETAS3D":
        """Replace patch productivity weights and refresh per-group totals."""

        value = torch.as_tensor(
            tau, dtype=self.patch_tau.dtype, device=self.patch_tau.device
        )
        self._validate_patch_vector(value, "patch_tau", nonnegative=True)
        value = value.clone()
        totals = self._group_totals(value)
        zero_groups = torch.nonzero(totals <= 0.0, as_tuple=False).flatten()
        for ordinal_tensor in zero_groups:
            ordinal = int(ordinal_tensor.item())
            begin = int(self.finite_source_group_offsets[ordinal].item())
            end = int(self.finite_source_group_offsets[ordinal + 1].item())
            if self.zero_responsibility_policy == "retain":
                value[begin:end] = self.patch_tau[begin:end]
            else:
                value[begin:end] = 1.0
        totals = self._group_totals(value)
        if bool(torch.any(totals <= 0.0)):
            raise ValueError("patch_tau must have positive mass in every group")
        self.patch_tau.copy_(value)
        self.patch_tau_total.copy_(totals)
        return self

    update_patch_tau_ = set_patch_tau_

    @torch.no_grad()
    def set_patch_smoothing_normalizer_(
        self, normalizer: torch.Tensor
    ) -> "FiniteSourceETAS3D":
        value = torch.as_tensor(
            normalizer,
            dtype=self.patch_smoothing_normalizer.dtype,
            device=self.patch_smoothing_normalizer.device,
        )
        self._validate_patch_vector(
            value, "patch_smoothing_normalizer", positive=True
        )
        self.patch_smoothing_normalizer.copy_(value)
        return self

    def patch_slice(self, group: int | torch.Tensor) -> slice:
        """Return the global contiguous patch slice for a source-group ID."""

        group_id = int(group.item()) if isinstance(group, torch.Tensor) else int(group)
        try:
            return self._group_slices[group_id]
        except KeyError as error:
            raise KeyError(f"finite source ID {group_id} has no patches") from error

    def finite_source_parent_indices(self, catalog: Any) -> torch.Tensor:
        """Return catalog row indices whose finite-source ID is positive."""

        return self._source_partition(catalog)[0]

    def _source_partition(self, catalog):
        ids = getattr(catalog, "finite_source_id", None)
        if ids is None:
            return self._checked_parent_indices(catalog), None, ()
        if not isinstance(ids, torch.Tensor) or ids.shape != catalog.time.shape or ids.dtype != torch.int64:
            self._checked_parent_indices(catalog)
        cache = self.__dict__.setdefault("_partition_cache", FixedInputCache())
        def build():
            parents = self._checked_parent_indices(catalog)
            points = torch.nonzero(ids == 0, as_tuple=False).flatten()
            rows = parents.detach().cpu().tolist()
            groups = ids[parents].detach().cpu().tolist()
            return parents, points, tuple(zip(rows, groups))
        return cache.get_or_create((ids, self.finite_source_group_ids), (), build)

    def _checked_parent_indices(self, catalog: Any) -> torch.Tensor:

        source_ids = getattr(catalog, "finite_source_id", None)
        if source_ids is None:
            return torch.empty(0, dtype=torch.int64, device=catalog.time.device)
        if not isinstance(source_ids, torch.Tensor):
            raise TypeError("catalog.finite_source_id must be a tensor or None")
        if source_ids.shape != catalog.time.shape:
            raise ValueError("catalog.finite_source_id must have shape [N]")
        if source_ids.dtype != torch.int64:
            raise TypeError("catalog.finite_source_id must have dtype torch.int64")
        if bool(torch.any(source_ids < 0)):
            raise ValueError("catalog.finite_source_id values must be non-negative")
        parent_indices = torch.nonzero(source_ids > 0, as_tuple=False).flatten()
        if parent_indices.numel() == 0:
            return parent_indices
        parent_ids = source_ids[parent_indices]
        if torch.unique(parent_ids).numel() != parent_ids.numel():
            raise ValueError(
                "each positive finite_source_id must identify one parent event"
            )
        available = self.finite_source_group_ids
        positions = torch.searchsorted(available, parent_ids)
        safe_positions = positions.clamp_max(max(available.numel() - 1, 0))
        if bool(
            torch.any(positions >= available.numel())
            or torch.any(available[safe_positions] != parent_ids)
        ):
            missing = parent_ids[
                (positions >= available.numel())
                | (available[safe_positions] != parent_ids)
            ]
            raise KeyError(
                "catalog references finite-source IDs without patches: "
                f"{missing.detach().cpu().tolist()}"
            )
        return parent_indices

    @staticmethod
    def _take(data: Any, indices: torch.Tensor) -> Any:
        if hasattr(data, "take"):
            return data.take(indices)
        raise TypeError("source must provide take(indices)")

    def _patch_source(self, parent: Any, begin: int, end: int) -> SimpleNamespace:
        if parent.time.shape != (1,):
            raise ValueError("finite-source parent must contain exactly one event")
        patch_count = end - begin
        if patch_count < 1:
            raise ValueError("a finite patch block cannot be empty")
        values = dict(
            time=parent.time.expand(patch_count),
            position=self.patch_position[begin:end],
            depth=self.patch_depth[begin:end],
            magnitude=parent.magnitude.expand(patch_count),
            completeness_magnitude=parent.completeness_magnitude,
            finite_source_id=self.patch_finite_source_id[begin:end],
        )
        parent_order = getattr(parent, "sequence_index", None)
        if parent_order is not None:
            values["sequence_index"] = parent_order.expand(patch_count)
        return SimpleNamespace(**values)

    def spatial_scale(self, source: Any) -> torch.Tensor:
        point_scale = super().spatial_scale(source)
        source_ids = getattr(source, "finite_source_id", None)
        if source_ids is None:
            return point_scale
        return torch.where(source_ids > 0, self.chara_d, point_scale)

    def depth_concentration(self, source: Any) -> torch.Tensor:
        point_concentration = super().depth_concentration(source)
        source_ids = getattr(source, "finite_source_id", None)
        if source_ids is None:
            return point_concentration
        return torch.where(source_ids > 0, self.ch_eta, point_concentration)

    def finite_patch_rate(
        self,
        target: Any,
        parent: Any,
        begin: int,
        end: int,
    ) -> torch.Tensor:
        """Return weighted rates from one parent's requested patch block."""

        self._validate_target(target)
        self._validate_source(parent)
        self._validate_depth(target, "target")
        self._validate_depth(parent, "source")
        if not 0 <= begin < end <= self.patch_count:
            raise IndexError("patch bounds must satisfy 0 <= begin < end <= P")
        parent_ids = getattr(parent, "finite_source_id", None)
        if parent_ids is None or parent_ids.shape != (1,):
            raise ValueError(
                "finite-source parent must provide finite_source_id with shape [1]"
            )
        parent_groups = self._source_partition(parent)[2]
        group_id = parent_groups[0][1] if parent_groups else 0
        group_slice = self.patch_slice(group_id)
        if begin < group_slice.start or end > group_slice.stop:
            raise ValueError("patch block does not belong to the supplied parent")

        patch_source = self._patch_source(parent, begin, end)
        delta_time = target.time[:, None] - patch_source.time[None, :]
        displacement = (
            target.position[:, None, :] - patch_source.position[None, :, :]
        )
        distance_squared = displacement.square().sum(dim=-1)
        productivity = self.magnitude_productivity(patch_source)[None, :]
        target_order = getattr(target, "sequence_index", None)
        source_order = getattr(patch_source, "sequence_index", None)
        causal_mask = None
        if target_order is not None or source_order is not None:
            if target_order is None or source_order is None:
                raise ValueError(
                    "target and finite source must both provide sequence_index or neither"
                )
            causal_mask = source_order[None, :] < target_order[:, None]
        temporal = self.temporal_kernel(delta_time, causal_mask)
        spatial = self.spatial_kernel(distance_squared, patch_source)
        depth = conditional_beta_depth_density(
            target.depth[:, None],
            patch_source.depth[None, :],
            self.maximum_depth,
            self.ch_eta,
            boundary_offset=self.depth_boundary_offset,
        )
        ordinal = self._group_ordinals[group_id]
        weights = self.patch_tau[begin:end] / self.patch_tau_total[ordinal]
        return productivity * temporal * spatial * depth * weights[None, :]

    def _checkpoint_contribution(self, function):
        """Local-only recomputation; EM states must remain fixed until backward."""
        if not self.checkpoint_patches or not torch.is_grad_enabled():
            return function()
        return checkpoint(function, use_reentrant=False)

    def pairwise_rate(self, target: Any, source: Any) -> torch.Tensor:
        """Return mixed triggering rates with shape [N_target, N_source]."""

        self._validate_target(target)
        self._validate_source(source)
        self._validate_depth(target, "target")
        self._validate_depth(source, "source")
        parent_indices, point_indices, parent_groups = self._source_partition(source)
        if parent_indices.numel() == 0:
            return self._checkpoint_contribution(
                lambda: super(FiniteSourceETAS3D, self).pairwise_rate(target, source))

        source_count = source.time.numel()
        result = target.time.new_zeros((target.time.numel(), source_count))
        source_ids = source.finite_source_id
        if point_indices.numel() > 0:
            point_source = self._take(source, point_indices)
            point_rate = self._checkpoint_contribution(
                lambda saved=point_source: super(FiniteSourceETAS3D, self).pairwise_rate(target, saved))
            result = result.index_add(1, point_indices, point_rate)

        for ordinal_index, (parent_index, group_id) in enumerate(parent_groups):
            parent_index_tensor = parent_indices[ordinal_index]
            parent = self._take(source, slice(parent_index, parent_index + 1))
            patch_range = self.patch_slice(group_id)
            parent_rate = target.time.new_zeros(target.time.numel())
            for begin in range(
                patch_range.start, patch_range.stop, self.patch_chunk_size
            ):
                end = min(begin + self.patch_chunk_size, patch_range.stop)
                parent_rate = parent_rate + self._checkpoint_contribution(
                    lambda saved_parent=parent, lo=begin, hi=end:
                    self.finite_patch_rate(target, saved_parent, lo, hi).sum(dim=-1))
            column = parent_index_tensor.reshape(1)
            result = result.index_add(1, column, parent_rate[:, None])
        return result

    def integrated_source(
        self,
        source: Any,
        domain: Any,
        space_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate point centroids or weighted finite patches over the domain."""

        self._validate_source(source)
        self._validate_depth(source, "source")
        parent_indices, point_indices, parent_groups = self._source_partition(source)
        if parent_indices.numel() == 0:
            return self._checkpoint_contribution(
                lambda: super(FiniteSourceETAS3D, self).integrated_source(source, domain, space_mass))
        if space_mass is not None:
            if space_mass.shape != source.time.shape:
                raise ValueError("space_mass must have shape [N_source]")
            time_mass = self.temporal_integral(
                source.time, domain.time_start, domain.time_end
            )
            return self.magnitude_productivity(source) * time_mass * space_mass

        result = source.time.new_zeros(source.time.shape)
        source_ids = source.finite_source_id
        if point_indices.numel() > 0:
            point_source = self._take(source, point_indices)
            point_integral = self._checkpoint_contribution(
                lambda saved=point_source: super(FiniteSourceETAS3D, self).integrated_source(saved, domain))
            result = result.index_add(0, point_indices, point_integral)

        for ordinal_index, (parent_index, group_id) in enumerate(parent_groups):
            parent_index_tensor = parent_indices[ordinal_index]
            parent = self._take(source, slice(parent_index, parent_index + 1))
            ordinal = self._group_ordinals[group_id]
            patch_range = self.patch_slice(group_id)
            parent_integral = source.time.new_zeros(())
            for begin in range(
                patch_range.start, patch_range.stop, self.patch_chunk_size
            ):
                end = min(begin + self.patch_chunk_size, patch_range.stop)
                patch_source = self._patch_source(parent, begin, end)
                def contribution(saved=patch_source, lo=begin, hi=end, group=ordinal):
                    patch_integral = super(FiniteSourceETAS3D, self).integrated_source(saved, domain)
                    weights = self.patch_tau[lo:hi] / self.patch_tau_total[group]
                    return (patch_integral * weights).sum()
                parent_integral = parent_integral + self._checkpoint_contribution(contribution)
            result = result.index_add(
                0, parent_index_tensor.reshape(1), parent_integral.reshape(1)
            )
        return result

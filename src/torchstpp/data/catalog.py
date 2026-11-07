"""Validated tensor representation of an event catalog."""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch


_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64)


def _check_float_dtype(dtype: torch.dtype) -> None:
    if dtype not in _SUPPORTED_FLOAT_DTYPES:
        raise TypeError("catalog dtype must be torch.float32 or torch.float64")


@dataclass(frozen=True, repr=False)
class EventCatalog:
    """A validated collection of events stored as tensors.

    ``position`` is deliberately limited to two columns in the first release so
    that a catalog can be passed directly to :class:`torchstpp.models.etas.ETAS2D`.
    Depth is retained separately for future 3-D models.  All floating-point
    tensors share one dtype and all tensors live on one device.

    The constructor validates but never sorts or alters observations.  File
    loaders perform a stable time sort before constructing this object.
    """

    event_id: torch.Tensor
    time: torch.Tensor
    position: torch.Tensor
    magnitude: torch.Tensor
    completeness_magnitude: torch.Tensor
    sequence_index: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    inside: torch.Tensor | None = None
    finite_source_id: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.sequence_index is None:
            object.__setattr__(
                self,
                "sequence_index",
                torch.arange(
                    self.time.shape[0], device=self.time.device, dtype=torch.int64
                ),
            )
        tensor_fields = {
            "event_id": self.event_id,
            "time": self.time,
            "position": self.position,
            "magnitude": self.magnitude,
            "completeness_magnitude": self.completeness_magnitude,
            "sequence_index": self.sequence_index,
        }
        for name, value in tensor_fields.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        event_count = self.time.shape[0] if self.time.ndim == 1 else -1
        if event_count < 1:
            raise ValueError("time must be a non-empty one-dimensional tensor")
        if self.event_id.shape != (event_count,):
            raise ValueError(f"event_id must have shape [{event_count}]")
        if self.sequence_index.shape != (event_count,):
            raise ValueError(f"sequence_index must have shape [{event_count}]")
        if self.position.shape != (event_count, 2):
            raise ValueError(f"position must have shape [{event_count}, 2]")
        if self.magnitude.shape != (event_count,):
            raise ValueError(f"magnitude must have shape [{event_count}]")
        if self.completeness_magnitude.ndim != 0:
            raise ValueError("completeness_magnitude must be a scalar tensor")
        if self.depth is not None:
            if not isinstance(self.depth, torch.Tensor):
                raise TypeError("depth must be a torch.Tensor or None")
            if self.depth.shape != (event_count,):
                raise ValueError(f"depth must have shape [{event_count}]")
        if self.inside is not None:
            if not isinstance(self.inside, torch.Tensor):
                raise TypeError("inside must be a torch.Tensor or None")
            if self.inside.shape != (event_count,):
                raise ValueError(f"inside must have shape [{event_count}]")
        if self.finite_source_id is not None:
            if not isinstance(self.finite_source_id, torch.Tensor):
                raise TypeError(
                    "finite_source_id must be a torch.Tensor or None"
                )
            if self.finite_source_id.shape != (event_count,):
                raise ValueError(
                    f"finite_source_id must have shape [{event_count}]"
                )

        if self.event_id.dtype != torch.int64:
            raise TypeError("event_id must have dtype torch.int64")
        if self.sequence_index.dtype != torch.int64:
            raise TypeError("sequence_index must have dtype torch.int64")
        if torch.unique(self.sequence_index).numel() != event_count:
            raise ValueError("sequence_index values must be unique")
        if event_count > 1 and not bool(
            torch.all(self.sequence_index[1:] > self.sequence_index[:-1])
        ):
            raise ValueError("sequence_index must preserve catalog row order")
        if self.inside is not None and self.inside.dtype != torch.bool:
            raise TypeError("inside must have dtype torch.bool")
        if (
            self.finite_source_id is not None
            and self.finite_source_id.dtype != torch.int64
        ):
            raise TypeError("finite_source_id must have dtype torch.int64")
        if self.finite_source_id is not None and bool(
            torch.any(self.finite_source_id < 0)
        ):
            raise ValueError("finite_source_id values must be nonnegative")

        _check_float_dtype(self.time.dtype)
        floating_tensors = {
            "position": self.position,
            "magnitude": self.magnitude,
            "completeness_magnitude": self.completeness_magnitude,
        }
        if self.depth is not None:
            floating_tensors["depth"] = self.depth
        for name, value in floating_tensors.items():
            if value.dtype != self.time.dtype:
                raise TypeError(
                    f"{name} must have the same dtype as time ({self.time.dtype})"
                )

        all_tensors = dict(tensor_fields)
        if self.depth is not None:
            all_tensors["depth"] = self.depth
        if self.inside is not None:
            all_tensors["inside"] = self.inside
        if self.finite_source_id is not None:
            all_tensors["finite_source_id"] = self.finite_source_id
        for name, value in all_tensors.items():
            if value.device != self.time.device:
                raise ValueError(
                    f"{name} must be on the same device as time ({self.time.device})"
                )

        finite_tensors = {
            "time": self.time,
            "position": self.position,
            "magnitude": self.magnitude,
            "completeness_magnitude": self.completeness_magnitude,
        }
        if self.depth is not None:
            finite_tensors["depth"] = self.depth
        for name, value in finite_tensors.items():
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{name} contains NaN or infinite values")

        if torch.unique(self.event_id).numel() != event_count:
            raise ValueError("event_id values must be unique")
        if event_count > 1 and not bool(torch.all(self.time[1:] >= self.time[:-1])):
            raise ValueError("time must be sorted in non-decreasing order")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        # Keep a private deep snapshot rather than MappingProxyType: plain mappings
        # remain compatible with pickle, torch.save, and multiprocessing spawn.
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))

    def __len__(self) -> int:
        return self.time.shape[0]

    def __repr__(self) -> str:
        return (
            f"EventCatalog(events={len(self)}, device='{self.device}', "
            f"dtype={self.dtype}, depth={self.depth is not None}, "
            f"inside={self.inside is not None}, "
            f"finite_source={self.finite_source_id is not None})"
        )

    @property
    def device(self) -> torch.device:
        return self.time.device

    @property
    def dtype(self) -> torch.dtype:
        return self.time.dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
    ) -> "EventCatalog":
        """Return a new catalog on ``device`` and in floating ``dtype``.

        Integer IDs remain ``int64`` and the optional region mask remains
        boolean.  An explicit CUDA request fails rather than silently falling
        back to the CPU.
        """

        target_device = self.device if device is None else torch.device(device)
        target_dtype = self.dtype if dtype is None else dtype
        _check_float_dtype(target_dtype)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{target_device}' was requested, but CUDA is not available"
            )

        float_kwargs = {
            "device": target_device,
            "dtype": target_dtype,
            "non_blocking": non_blocking,
        }
        device_kwargs = {"device": target_device, "non_blocking": non_blocking}
        metadata = copy.deepcopy(dict(self.metadata))
        metadata.update(
            {
                "tensor_dtype": str(target_dtype),
                "tensor_device": str(target_device),
            }
        )
        return EventCatalog(
            event_id=self.event_id.to(**device_kwargs),
            sequence_index=self.sequence_index.to(**device_kwargs),
            time=self.time.to(**float_kwargs),
            position=self.position.to(**float_kwargs),
            magnitude=self.magnitude.to(**float_kwargs),
            completeness_magnitude=self.completeness_magnitude.to(**float_kwargs),
            depth=None if self.depth is None else self.depth.to(**float_kwargs),
            inside=None if self.inside is None else self.inside.to(**device_kwargs),
            finite_source_id=None
            if self.finite_source_id is None
            else self.finite_source_id.to(**device_kwargs),
            metadata=metadata,
        )

    def cpu(self) -> "EventCatalog":
        return self.to(device="cpu")

    def cuda(self, device: torch.device | str = "cuda:0") -> "EventCatalog":
        return self.to(device=device)

    def take(
        self,
        selector: slice | torch.Tensor | Sequence[int],
    ) -> "EventCatalog":
        """Select events with a slice, boolean mask, or integer indices.

        This is the canonical selection interface used by likelihood,
        inference, event-state, and relation modules.  Contiguous slices reuse
        :meth:`slice`; boolean masks and general integer selections reuse
        :meth:`index_select`.

        The catalog invariant requires at least one event, so selectors that
        select no events raise ``ValueError``.
        """

        if isinstance(selector, slice):
            start, stop, step = selector.indices(len(self))
            if step == 1:
                if start >= stop:
                    raise ValueError("selector cannot select zero events")
                return self.slice(start, stop)
            indices = torch.arange(
                start,
                stop,
                step,
                dtype=torch.int64,
                device=self.device,
            )
            if indices.numel() == 0:
                raise ValueError("selector cannot select zero events")
            return self.index_select(indices)

        index = torch.as_tensor(selector, device=self.device)
        if index.ndim != 1:
            raise ValueError("selector must be one-dimensional")
        if index.dtype == torch.bool:
            if index.shape != (len(self),):
                raise ValueError(
                    f"boolean selector must have shape [{len(self)}]"
                )
            index = torch.nonzero(index, as_tuple=False).flatten()
            if index.numel() == 0:
                raise ValueError("selector cannot select zero events")
        return self.index_select(index)

    def filter_for_analysis(
        self,
        *,
        time_zero: float | torch.Tensor | None = None,
        time_end: float | torch.Tensor | None = None,
    ) -> "EventCatalog":
        """Apply the explicit ETAS source-window and completeness filters.

        Events must satisfy ``magnitude >= completeness_magnitude`` and
        ``time_zero <= time <= time_end``.  Omitting a time bound uses the
        corresponding catalog bound.  Original ``event_id`` values are
        preserved, and the applied bounds are recorded in metadata.
        """

        resolved_zero = self.time[0] if time_zero is None else torch.as_tensor(
            time_zero, dtype=self.dtype, device=self.device
        )
        resolved_end = self.time[-1] if time_end is None else torch.as_tensor(
            time_end, dtype=self.dtype, device=self.device
        )
        if resolved_zero.ndim != 0 or resolved_end.ndim != 0:
            raise ValueError("time_zero and time_end must be scalars")
        if not bool(torch.isfinite(resolved_zero)) or not bool(
            torch.isfinite(resolved_end)
        ):
            raise ValueError("time_zero and time_end must be finite")
        if not bool(resolved_end >= resolved_zero):
            raise ValueError("time_end must not be earlier than time_zero")

        mask = (
            (self.magnitude >= self.completeness_magnitude)
            & (self.time >= resolved_zero)
            & (self.time <= resolved_end)
        )
        selected_count = int(mask.sum().item())
        if selected_count == 0:
            raise ValueError(
                "analysis filters select no events; check time_zero, time_end, "
                "and completeness_magnitude"
            )
        selected = self.take(mask)
        metadata = copy.deepcopy(dict(selected.metadata))
        metadata.update(
            {
                "analysis_filter_applied": True,
                "analysis_time_zero": float(resolved_zero.item()),
                "analysis_time_end": float(resolved_end.item()),
                "analysis_minimum_magnitude": float(
                    self.completeness_magnitude.item()
                ),
                "analysis_input_event_count": len(self),
                "analysis_event_count": selected_count,
            }
        )
        return replace(selected, metadata=metadata)

    def resolve_duplicates(
        self,
        *,
        time_policy: str = "jitter",
        position_policy: str = "jitter",
        position_minimum_separation_m: float = 0.05,
        depth_policy: str = "keep",
        depth_minimum_separation: float = 0.05,
        warn: bool = True,
    ) -> "EventCatalog":
        """Resolve finite-precision ties in time, horizontal position, and depth.

        ``jitter`` is deterministic. Tied times are advanced with
        :func:`torch.nextafter`, which is the smallest increment representable
        by the catalog dtype. Coincident positions use the legacy golden-angle
        displacement with the first event left unchanged. Repeated depth
        values are preserved by default because earthquake depths are commonly
        rounded and equal depths alone do not identify duplicate events. Set
        ``depth_policy="jitter"`` only for an experiment that explicitly
        requires distinct depth coordinates. ``keep`` preserves the input and
        ``error`` rejects it.
        """

        valid_policies = {"jitter", "keep", "error"}
        if time_policy not in valid_policies:
            raise ValueError("time_policy must be 'jitter', 'keep', or 'error'")
        if position_policy not in valid_policies:
            raise ValueError(
                "position_policy must be 'jitter', 'keep', or 'error'"
            )
        if depth_policy not in valid_policies:
            raise ValueError("depth_policy must be 'jitter', 'keep', or 'error'")
        separation_m = float(position_minimum_separation_m)
        if not math.isfinite(separation_m) or separation_m <= 0.0:
            raise ValueError(
                "position_minimum_separation_m must be finite and positive"
            )
        depth_separation = float(depth_minimum_separation)
        if not math.isfinite(depth_separation) or depth_separation <= 0.0:
            raise ValueError(
                "depth_minimum_separation must be finite and positive"
            )

        time = self.time
        tied_time_pairs = int((time[1:] <= time[:-1]).sum().item())
        time_modified = 0
        if tied_time_pairs and time_policy == "error":
            raise ValueError(
                f"catalog contains {tied_time_pairs} adjacent finite-precision "
                "time ties"
            )
        if tied_time_pairs and time_policy == "jitter":
            time = time.clone()
            infinity = time.new_tensor(float("inf"))
            for index in range(1, len(self)):
                if bool(time[index] <= time[index - 1]):
                    adjusted = torch.nextafter(time[index - 1], infinity)
                    if not bool(torch.isfinite(adjusted)):
                        raise FloatingPointError(
                            "cannot make catalog times strictly increasing in "
                            f"{self.dtype}"
                        )
                    time[index] = adjusted
                    time_modified += 1

        unique_position, inverse, counts = torch.unique(
            self.position, dim=0, return_inverse=True, return_counts=True
        )
        duplicate_groups = torch.nonzero(counts > 1, as_tuple=False).flatten()
        duplicate_position_events = int((counts[counts > 1] - 1).sum().item())
        if duplicate_position_events and position_policy == "error":
            raise ValueError(
                f"catalog contains {duplicate_position_events} events beyond "
                "the first at exactly repeated positions"
            )
        position = self.position
        position_modified = 0
        if duplicate_position_events and position_policy == "jitter":
            position = position.clone()
            coordinate_kind = str(
                self.metadata.get("coordinate_kind", "")
            ).lower()
            distance_unit = str(self.metadata.get("distance_unit", "")).lower()
            geographic = coordinate_kind == "geographic" or distance_unit in {
                "degree", "degrees", "deg"
            }
            unit_metres = {
                "m": 1.0,
                "meter": 1.0,
                "meters": 1.0,
                "metre": 1.0,
                "metres": 1.0,
                "km": 1000.0,
                "kilometer": 1000.0,
                "kilometers": 1000.0,
                "kilometre": 1000.0,
                "kilometres": 1000.0,
            }
            if not geographic and distance_unit not in unit_metres:
                raise ValueError(
                    "position jitter requires geographic coordinates or a "
                    "distance unit of metres/kilometres; use position_policy="
                    "'keep' or identify the catalog coordinate units"
                )
            golden_angle = 2.399963229728653
            for group in duplicate_groups:
                indices = torch.nonzero(
                    inverse == group, as_tuple=False
                ).flatten()
                ordinal = torch.arange(
                    indices.numel(), dtype=self.dtype, device=self.device
                )
                radius_m = ordinal.sqrt() * separation_m
                angle = ordinal * golden_angle
                east_m = radius_m * torch.cos(angle)
                north_m = radius_m * torch.sin(angle)
                if geographic:
                    latitude = unique_position[group, 1]
                    cos_latitude = torch.cos(torch.deg2rad(latitude)).abs().clamp_min(
                        1e-6
                    )
                    dx = east_m / (111_320.0 * cos_latitude)
                    dy = north_m / 111_320.0
                else:
                    dx = east_m / unit_metres[distance_unit]
                    dy = north_m / unit_metres[distance_unit]
                candidate = torch.stack(
                    (
                        unique_position[group, 0] + dx,
                        unique_position[group, 1] + dy,
                    ),
                    dim=1,
                )
                # A sub-metre displacement may round back to the same degree
                # coordinate in float32. Advance x by one ULP until every
                # member of the duplicate group is actually distinguishable.
                for local_index in range(1, indices.numel()):
                    direction = (
                        float("inf")
                        if bool(torch.cos(angle[local_index]) >= 0)
                        else float("-inf")
                    )
                    target = candidate.new_tensor(direction)
                    while bool(
                        torch.any(
                            torch.all(
                                candidate[:local_index]
                                == candidate[local_index],
                                dim=1,
                            )
                        )
                    ):
                        candidate[local_index, 0] = torch.nextafter(
                            candidate[local_index, 0], target
                        )
                position[indices] = candidate
                position_modified += indices.numel() - 1

        depth = self.depth
        depth_duplicate_groups = self.event_id.new_empty((0,))
        duplicate_depth_events = 0
        depth_modified = 0
        if depth is not None:
            unique_depth, depth_inverse, depth_counts = torch.unique(
                depth, return_inverse=True, return_counts=True
            )
            depth_duplicate_groups = torch.nonzero(
                depth_counts > 1, as_tuple=False
            ).flatten()
            duplicate_depth_events = int(
                (depth_counts[depth_counts > 1] - 1).sum().item()
            )
            if duplicate_depth_events and depth_policy == "error":
                raise ValueError(
                    f"catalog contains {duplicate_depth_events} events beyond "
                    "the first at exactly repeated depths"
                )
            if duplicate_depth_events and depth_policy == "jitter":
                depth = depth.clone()
                # Catalog depths are commonly rounded to a regular metre grid.
                # An exact multiple of the requested separation could land on
                # another occupied grid point, so use a tiny golden-ratio
                # inflation while preserving the requested physical scale.
                golden_ratio_fraction = 0.6180339887498949
                effective_depth_separation = depth_separation * (
                    1.0 + 1e-3 * golden_ratio_fraction
                )
                for group in depth_duplicate_groups:
                    indices = torch.nonzero(
                        depth_inverse == group, as_tuple=False
                    ).flatten()
                    ordinal = torch.arange(
                        indices.numel(), dtype=self.dtype, device=self.device
                    )
                    steps = torch.floor((ordinal + 1.0) / 2.0)
                    signs = torch.where(
                        torch.remainder(ordinal, 2.0) == 1.0,
                        torch.ones_like(ordinal),
                        -torch.ones_like(ordinal),
                    )
                    offsets = steps * signs * effective_depth_separation
                    candidate = unique_depth[group] + offsets
                    # Avoid introducing negative positive-down depths near the
                    # surface; use one-sided offsets for those rare groups.
                    if bool(torch.any(candidate < 0.0)):
                        candidate = (
                            unique_depth[group]
                            + ordinal * effective_depth_separation
                        )
                    for local_index in range(1, indices.numel()):
                        direction = (
                            float("inf")
                            if bool(candidate[local_index] >= unique_depth[group])
                            else float("-inf")
                        )
                        target = candidate.new_tensor(direction)
                        while bool(
                            torch.any(candidate[:local_index] == candidate[local_index])
                        ):
                            candidate[local_index] = torch.nextafter(
                                candidate[local_index], target
                            )
                    depth[indices] = candidate
                    depth_modified += indices.numel() - 1

        if warn and time_modified:
            warnings.warn(
                f"adjusted {time_modified} event times by the minimum "
                f"representable {self.dtype} increments; set time duplicate "
                "policy to 'keep' or 'error' to change this behavior",
                UserWarning,
                stacklevel=2,
            )
        if warn and position_modified:
            warnings.warn(
                f"jittered {position_modified} events at repeated positions "
                f"using {separation_m:g} m legacy separation; set position "
                "duplicate policy to 'keep' or 'error' to change this behavior",
                UserWarning,
                stacklevel=2,
            )
        if warn and depth_modified:
            depth_unit = str(self.metadata.get("depth_unit", "catalog units"))
            warnings.warn(
                f"jittered {depth_modified} events at repeated depths using "
                f"{depth_separation:g} {depth_unit} minimum separation; set "
                "depth duplicate policy to 'keep' or 'error' to change this "
                "behavior",
                UserWarning,
                stacklevel=2,
            )

        metadata = copy.deepcopy(dict(self.metadata))
        metadata["duplicate_resolution"] = {
            "time_policy": time_policy,
            "time_tied_adjacent_pairs": tied_time_pairs,
            "time_modified_events": time_modified,
            "time_increment": "nextafter" if time_policy == "jitter" else "none",
            "time_dtype": str(self.dtype).removeprefix("torch."),
            "position_policy": position_policy,
            "position_duplicate_groups": int(duplicate_groups.numel()),
            "position_modified_events": position_modified,
            "position_minimum_separation_m": separation_m,
            "position_increment": "golden_angle_with_nextafter_fallback",
            "depth_policy": depth_policy,
            "depth_duplicate_groups": int(depth_duplicate_groups.numel()),
            "depth_modified_events": depth_modified,
            "depth_minimum_separation": depth_separation,
            "depth_increment": "symmetric_golden_offset_with_nextafter_fallback",
        }
        return replace(
            self, time=time, position=position, depth=depth, metadata=metadata
        )

    def project(self, transformer: Any) -> "EventCatalog":
        """Return this catalog in the transformer's projected coordinates."""

        if not hasattr(transformer, "forward_catalog"):
            raise TypeError("transformer must provide forward_catalog(catalog)")
        return transformer.forward_catalog(self)

    def inverse_project(self, transformer: Any) -> "EventCatalog":
        """Return a projected catalog in the transformer's source coordinates."""

        if not hasattr(transformer, "inverse_catalog"):
            raise TypeError("transformer must provide inverse_catalog(catalog)")
        return transformer.inverse_catalog(self)

    def slice(self, start: int, end: int) -> "EventCatalog":
        """Return a contiguous event block without changing global IDs."""

        if not 0 <= start < end <= len(self):
            raise IndexError(
                f"slice bounds must satisfy 0 <= start < end <= {len(self)}"
            )
        if start == 0 and end == len(self):
            return self
        item = slice(start, end)
        return EventCatalog(
            event_id=self.event_id[item],
            sequence_index=self.sequence_index[item],
            time=self.time[item],
            position=self.position[item],
            magnitude=self.magnitude[item],
            completeness_magnitude=self.completeness_magnitude,
            depth=None if self.depth is None else self.depth[item],
            inside=None if self.inside is None else self.inside[item],
            finite_source_id=None
            if self.finite_source_id is None
            else self.finite_source_id[item],
            metadata=self.metadata,
        )

    def index_select(self, indices: torch.Tensor | Sequence[int]) -> "EventCatalog":
        """Select events while preserving their global event IDs."""

        index = torch.as_tensor(indices, device=self.device)
        if index.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        if index.dtype == torch.bool or index.is_floating_point():
            raise TypeError("indices must contain integers, not booleans or floats")
        index = index.to(dtype=torch.int64)
        if index.numel() < 1:
            raise ValueError("indices cannot be empty")
        if bool(torch.any(index < 0)) or bool(torch.any(index >= len(self))):
            raise IndexError("indices contain an out-of-range event position")
        selected_time = torch.index_select(self.time, 0, index)
        if selected_time.numel() > 1 and not bool(
            torch.all(selected_time[1:] >= selected_time[:-1])
        ):
            raise ValueError("indices must preserve non-decreasing time order")

        def select_optional(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else torch.index_select(value, 0, index)

        return EventCatalog(
            event_id=torch.index_select(self.event_id, 0, index),
            sequence_index=torch.index_select(self.sequence_index, 0, index),
            time=selected_time,
            position=torch.index_select(self.position, 0, index),
            magnitude=torch.index_select(self.magnitude, 0, index),
            completeness_magnitude=self.completeness_magnitude,
            depth=select_optional(self.depth),
            inside=select_optional(self.inside),
            finite_source_id=select_optional(self.finite_source_id),
            metadata=self.metadata,
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact, JSON-serializable catalog summary."""

        tensor_bytes = sum(
            value.numel() * value.element_size()
            for value in (
                self.event_id,
                self.sequence_index,
                self.time,
                self.position,
                self.magnitude,
                self.completeness_magnitude,
                self.depth,
                self.inside,
                self.finite_source_id,
            )
            if value is not None
        )
        result: dict[str, Any] = {
            "events": len(self),
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "tensor_bytes": tensor_bytes,
            "time_range": [float(self.time[0].item()), float(self.time[-1].item())],
            "position_min": self.position.amin(dim=0).detach().cpu().tolist(),
            "position_max": self.position.amax(dim=0).detach().cpu().tolist(),
            "magnitude_range": [
                float(self.magnitude.amin().item()),
                float(self.magnitude.amax().item()),
            ],
            "completeness_magnitude": float(
                self.completeness_magnitude.item()
            ),
            "has_depth": self.depth is not None,
            "has_inside": self.inside is not None,
            "has_finite_source_id": self.finite_source_id is not None,
        }
        if self.depth is not None:
            result["depth_range"] = [
                float(self.depth.amin().item()),
                float(self.depth.amax().item()),
            ]
        if self.inside is not None:
            result["inside_events"] = int(self.inside.sum().item())
        if self.finite_source_id is not None:
            finite_mask = self.finite_source_id > 0
            result["finite_source_events"] = int(finite_mask.sum().item())
            result["finite_source_groups"] = int(
                torch.unique(self.finite_source_id[finite_mask]).numel()
            )
        if "duplicate_resolution" in self.metadata:
            result["duplicate_resolution"] = copy.deepcopy(
                self.metadata["duplicate_resolution"]
            )
        return result

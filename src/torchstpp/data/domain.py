"""Observation time windows and planar spatial domains."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch

from .catalog import EventCatalog


_SUPPORTED_FLOAT_DTYPES = (torch.float32, torch.float64)
_SpatialKind = Literal["bbox", "polygon"]


@dataclass(frozen=True, repr=False)
class ObservationDomain:
    """A resolved observation interval and simple planar polygon.

    ``time_zero`` starts the source-history window, while ``time_start`` and
    ``time_end`` delimit likelihood targets.  ``border`` limits targets in the
    horizontal plane; source events are not spatially clipped.

    A domain selects *target* events for the likelihood.  Calling
    :meth:`select_events` must not be used to discard history or out-of-domain
    events that should remain available as possible triggering sources.
    """

    time_zero: torch.Tensor
    time_start: torch.Tensor
    time_end: torch.Tensor
    border: torch.Tensor
    spatial_kind: _SpatialKind = "polygon"
    include_start: bool = True
    include_end: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("time_zero", self.time_zero),
            ("time_start", self.time_start),
            ("time_end", self.time_end),
            ("border", self.border),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
        if (
            self.time_zero.ndim != 0
            or self.time_start.ndim != 0
            or self.time_end.ndim != 0
        ):
            raise ValueError("domain time bounds must be scalar tensors")
        if self.border.ndim != 2 or self.border.shape[1] != 2:
            raise ValueError("border must have shape [N_vertices, 2]")
        if self.spatial_kind not in {"bbox", "polygon"}:
            raise ValueError("spatial_kind must be 'bbox' or 'polygon'")
        if not isinstance(self.include_start, bool) or not isinstance(
            self.include_end, bool
        ):
            raise TypeError("include_start and include_end must be booleans")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        dtype = self.time_zero.dtype
        device = self.time_zero.device
        if dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError("domain dtype must be torch.float32 or torch.float64")
        for name, value in (
            ("time_start", self.time_start),
            ("time_end", self.time_end),
            ("border", self.border),
        ):
            if value.dtype != dtype:
                raise TypeError(f"{name} must have the same dtype as time_zero")
            if value.device != device:
                raise ValueError(f"{name} must be on the same device as time_zero")
        if not all(
            bool(torch.isfinite(value))
            for value in (self.time_zero, self.time_start, self.time_end)
        ):
            raise ValueError("domain time bounds must be finite")
        if not bool(self.time_start >= self.time_zero):
            raise ValueError("time_start must not be earlier than time_zero")
        if not bool(self.time_end > self.time_start):
            raise ValueError("time_end must be greater than time_start")
        if not bool(torch.all(torch.isfinite(self.border))):
            raise ValueError("border contains NaN or infinite values")

        border = _normalize_polygon(self.border)
        if self.spatial_kind == "bbox":
            if border.shape[0] != 4:
                raise ValueError("a bbox domain must contain exactly four vertices")
            lower = border.amin(dim=0)
            upper = border.amax(dim=0)
            expected = _bbox_border(lower[0], lower[1], upper[0], upper[1])
            if not bool(torch.equal(border, expected)):
                raise ValueError(
                    "a bbox domain must be an axis-aligned rectangle; use "
                    "spatial_kind='polygon' for a general quadrilateral"
                )
        object.__setattr__(self, "border", border)
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))

    @classmethod
    def from_catalog(
        cls,
        catalog: EventCatalog,
        *,
        time_zero: float | torch.Tensor | None = None,
        time_start: float | torch.Tensor | None = None,
        time_end: float | torch.Tensor | None = None,
        bbox: Sequence[float] | torch.Tensor | None = None,
        polygon: Sequence[Sequence[float]] | torch.Tensor | None = None,
        padding: float | Sequence[float] | torch.Tensor = 0.0,
        include_start: bool = True,
        include_end: bool = True,
    ) -> "ObservationDomain":
        """Resolve explicit bounds and infer an omitted interval or region.

        ``bbox`` uses ``(x_min, y_min, x_max, y_max)`` and must be expressed in
        the same projected coordinate system as ``catalog.position``.  A
        polygon is an ``[N_vertices, 2]`` sequence.  ``bbox`` and ``polygon``
        are mutually exclusive.

        If the spatial domain is omitted, a bounding rectangle is inferred
        from events inside the resolved time interval.  ``padding`` (one value
        or ``(x_padding, y_padding)``) is only valid for that automatic bbox.
        """

        return resolve_domain(
            catalog,
            time_zero=time_zero,
            time_start=time_start,
            time_end=time_end,
            bbox=bbox,
            polygon=polygon,
            padding=padding,
            include_start=include_start,
            include_end=include_end,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.time_start.dtype

    @property
    def device(self) -> torch.device:
        return self.time_start.device

    @property
    def duration(self) -> torch.Tensor:
        return self.time_end - self.time_start

    @property
    def history_duration(self) -> torch.Tensor:
        return self.time_end - self.time_zero

    @property
    def area(self) -> torch.Tensor:
        return torch.abs(_signed_twice_area(self.border)) / 2.0

    @property
    def spacetime_measure(self) -> torch.Tensor:
        return self.duration * self.area

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
    ) -> "ObservationDomain":
        """Return the domain on another device or floating dtype."""

        target_device = self.device if device is None else torch.device(device)
        target_dtype = self.dtype if dtype is None else dtype
        if target_dtype not in _SUPPORTED_FLOAT_DTYPES:
            raise TypeError("domain dtype must be torch.float32 or torch.float64")
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{target_device}' was requested, but CUDA is not available"
            )
        kwargs = {
            "device": target_device,
            "dtype": target_dtype,
            "non_blocking": non_blocking,
        }
        metadata = copy.deepcopy(dict(self.metadata))
        metadata.update(
            {
                "tensor_dtype": str(target_dtype),
                "tensor_device": str(target_device),
            }
        )
        return ObservationDomain(
            time_zero=self.time_zero.to(**kwargs),
            time_start=self.time_start.to(**kwargs),
            time_end=self.time_end.to(**kwargs),
            border=self.border.to(**kwargs),
            spatial_kind=self.spatial_kind,
            include_start=self.include_start,
            include_end=self.include_end,
            metadata=metadata,
        )

    def time_mask(self, catalog: EventCatalog) -> torch.Tensor:
        """Return target events inside ``[time_start, time_end]``."""

        self._validate_catalog_compatibility(catalog)
        lower = (
            catalog.time >= self.time_start
            if self.include_start
            else catalog.time > self.time_start
        )
        upper = (
            catalog.time <= self.time_end
            if self.include_end
            else catalog.time < self.time_end
        )
        return lower & upper

    def source_mask(self, catalog: EventCatalog) -> torch.Tensor:
        """Return possible source events inside ``[time_zero, time_end]``."""

        self._validate_catalog_compatibility(catalog)
        return (catalog.time >= self.time_zero) & (catalog.time <= self.time_end)

    def contains(self, position: torch.Tensor) -> torch.Tensor:
        """Return the spatial inclusion mask for planar positions."""

        if not isinstance(position, torch.Tensor):
            raise TypeError("position must be a torch.Tensor")
        if position.ndim != 2 or position.shape[1] != 2:
            raise ValueError("position must have shape [N, 2]")
        if position.device != self.device or position.dtype != self.dtype:
            raise TypeError("position and domain must share dtype and device")
        if self.spatial_kind == "bbox":
            lower = self.border.amin(dim=0)
            upper = self.border.amax(dim=0)
            return torch.all((position >= lower) & (position <= upper), dim=1)
        return _polygon_covers_points(self.border, position)

    def spatial_mask(self, catalog: EventCatalog) -> torch.Tensor:
        """Return catalog events covered by the planar polygon.

        Polygon edges and vertices count as inside, matching ``covers``
        semantics used by the earlier ETAS preprocessing code.
        """

        self._validate_catalog_compatibility(catalog)
        return self.contains(catalog.position)

    def event_mask(self, catalog: EventCatalog) -> torch.Tensor:
        """Return the target-event mask for the combined time-space domain."""

        return self.time_mask(catalog) & self.spatial_mask(catalog)

    def target_mask(self, catalog: EventCatalog) -> torch.Tensor:
        """Alias emphasizing that the domain selects likelihood targets."""

        return self.event_mask(catalog)

    def select_events(self, catalog: EventCatalog) -> EventCatalog:
        """Return target events without modifying or reordering the catalog."""

        indices = torch.nonzero(self.event_mask(catalog), as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("the observation domain contains no catalog events")
        return catalog.index_select(indices)

    def select_sources(self, catalog: EventCatalog) -> EventCatalog:
        """Return source-history events without spatial clipping."""

        mask = self.source_mask(catalog)
        if not bool(torch.any(mask)):
            raise ValueError("the source-history window contains no catalog events")
        return catalog.take(mask)

    def with_border(self, border: torch.Tensor, **metadata_updates: Any) -> "ObservationDomain":
        """Return the same time domain with a transformed polygon border."""

        metadata = copy.deepcopy(dict(self.metadata))
        metadata.update(metadata_updates)
        return ObservationDomain(
            time_zero=self.time_zero,
            time_start=self.time_start,
            time_end=self.time_end,
            border=border,
            spatial_kind="polygon",
            include_start=self.include_start,
            include_end=self.include_end,
            metadata=metadata,
        )

    def project(self, transformer: Any) -> "ObservationDomain":
        if not hasattr(transformer, "forward_domain"):
            raise TypeError("transformer must provide forward_domain(domain)")
        return transformer.forward_domain(self)

    def inverse_project(self, transformer: Any) -> "ObservationDomain":
        if not hasattr(transformer, "inverse_domain"):
            raise TypeError("transformer must provide inverse_domain(domain)")
        return transformer.inverse_domain(self)

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable description."""

        lower = self.border.amin(dim=0).detach().cpu().tolist()
        upper = self.border.amax(dim=0).detach().cpu().tolist()
        distance_unit = str(self.metadata.get("distance_unit", "unspecified"))
        return {
            "time_zero": float(self.time_zero.item()),
            "time_start": float(self.time_start.item()),
            "time_end": float(self.time_end.item()),
            "duration": float(self.duration.item()),
            "history_duration": float(self.history_duration.item()),
            "include_start": self.include_start,
            "include_end": self.include_end,
            "spatial_kind": self.spatial_kind,
            "bbox": [lower[0], lower[1], upper[0], upper[1]],
            "vertices": int(self.border.shape[0]),
            "area": float(self.area.item()),
            "time_unit": self.metadata.get("time_unit", "unspecified"),
            "distance_unit": distance_unit,
            "area_unit": (
                "unspecified"
                if distance_unit == "unspecified"
                else f"{distance_unit}^2"
            ),
            "coordinate_system": self.metadata.get(
                "coordinate_system", "unspecified"
            ),
            "time_start_inferred": bool(
                self.metadata.get("time_start_inferred", False)
            ),
            "time_end_inferred": bool(
                self.metadata.get("time_end_inferred", False)
            ),
            "spatial_inferred": bool(
                self.metadata.get("spatial_inferred", False)
            ),
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }

    def _validate_catalog_compatibility(self, catalog: EventCatalog) -> None:
        if not isinstance(catalog, EventCatalog):
            raise TypeError("catalog must be an EventCatalog")
        if catalog.device != self.device:
            raise ValueError(
                f"catalog is on {catalog.device}, but domain is on {self.device}; "
                "move them to the same device with .to(...)"
            )
        if catalog.dtype != self.dtype:
            raise TypeError(
                f"catalog has dtype {catalog.dtype}, but domain has dtype "
                f"{self.dtype}; move them to the same dtype with .to(...)"
            )
        for key, label in (
            ("time_unit", "time unit"),
            ("distance_unit", "distance unit"),
            ("coordinate_kind", "coordinate kind"),
            ("coordinate_system", "coordinate system"),
        ):
            domain_value = _normalized_metadata_value(self.metadata.get(key))
            catalog_value = _normalized_metadata_value(catalog.metadata.get(key))
            if (
                domain_value is not None
                and catalog_value is not None
                and domain_value != catalog_value
            ):
                raise ValueError(
                    f"domain {label} '{domain_value}' does not match catalog "
                    f"{label} '{catalog_value}'"
                )

    def __repr__(self) -> str:
        return (
            f"ObservationDomain(history=[{float(self.time_zero.item())}, "
            f"{float(self.time_end.item())}], targets=[{float(self.time_start.item())}, "
            f"{float(self.time_end.item())}], spatial_kind='{self.spatial_kind}', "
            f"vertices={self.border.shape[0]}, area={float(self.area.item()):.6g}, "
            f"device='{self.device}', dtype={self.dtype})"
        )


def resolve_domain(
    catalog: EventCatalog,
    *,
    time_zero: float | torch.Tensor | None = None,
    time_start: float | torch.Tensor | None = None,
    time_end: float | torch.Tensor | None = None,
    bbox: Sequence[float] | torch.Tensor | None = None,
    polygon: Sequence[Sequence[float]] | torch.Tensor | None = None,
    padding: float | Sequence[float] | torch.Tensor = 0.0,
    include_start: bool = True,
    include_end: bool = True,
) -> ObservationDomain:
    """Resolve a time-space domain from explicit and catalog-derived bounds.

    Automatic time bounds use the catalog minimum/maximum.  An automatic
    spatial bbox uses only events inside the resolved time interval.  This
    prevents prehistory sources from silently enlarging the target region.
    """

    if not isinstance(catalog, EventCatalog):
        raise TypeError("catalog must be an EventCatalog")
    if bbox is not None and polygon is not None:
        raise ValueError("bbox and polygon are mutually exclusive")
    if not isinstance(include_start, bool) or not isinstance(include_end, bool):
        raise TypeError("include_start and include_end must be booleans")
    _require_planar_catalog(catalog)

    zero_inferred = time_zero is None
    start_inferred = time_start is None
    end_inferred = time_end is None
    resolved_zero = (
        catalog.time.amin()
        if zero_inferred
        else _catalog_scalar(time_zero, catalog, "time_zero")
    )
    resolved_start = (
        catalog.time.amin()
        if start_inferred
        else _catalog_scalar(time_start, catalog, "time_start")
    )
    resolved_end = (
        catalog.time.amax()
        if end_inferred
        else _catalog_scalar(time_end, catalog, "time_end")
    )
    if not all(
        bool(torch.isfinite(value))
        for value in (resolved_zero, resolved_start, resolved_end)
    ):
        raise ValueError("domain time bounds must be finite")
    if not bool(resolved_start >= resolved_zero):
        raise ValueError("time_start must not be earlier than time_zero")
    if not bool(resolved_end > resolved_start):
        raise ValueError(
            "time_end must be greater than time_start; provide explicit bounds "
            "when the catalog cannot define a positive-duration interval"
        )

    lower_time = (
        catalog.time >= resolved_start
        if include_start
        else catalog.time > resolved_start
    )
    upper_time = (
        catalog.time <= resolved_end if include_end else catalog.time < resolved_end
    )
    candidate_mask = lower_time & upper_time
    candidate_count = int(candidate_mask.sum().item())

    padding_tensor = _resolve_padding(padding, catalog)
    explicit_spatial = bbox is not None or polygon is not None
    if explicit_spatial and bool(torch.any(padding_tensor != 0.0)):
        raise ValueError("padding is only valid when the spatial bbox is inferred")

    if polygon is not None:
        border = torch.as_tensor(
            polygon, dtype=catalog.dtype, device=catalog.device
        )
        spatial_kind: _SpatialKind = "polygon"
        spatial_inferred = False
    elif bbox is not None:
        bounds = torch.as_tensor(bbox, dtype=catalog.dtype, device=catalog.device)
        if bounds.ndim != 1 or bounds.numel() != 4:
            raise ValueError("bbox must contain (x_min, y_min, x_max, y_max)")
        if not bool(torch.all(torch.isfinite(bounds))):
            raise ValueError("bbox contains NaN or infinite values")
        x_min, y_min, x_max, y_max = bounds.unbind()
        if not bool(x_max > x_min) or not bool(y_max > y_min):
            raise ValueError("bbox must satisfy x_min < x_max and y_min < y_max")
        border = _bbox_border(x_min, y_min, x_max, y_max)
        spatial_kind = "bbox"
        spatial_inferred = False
    else:
        if candidate_count == 0:
            raise ValueError(
                "cannot infer a spatial bbox because the resolved time interval "
                "contains no events; specify bbox or polygon"
            )
        candidate_positions = catalog.position[candidate_mask]
        lower = candidate_positions.amin(dim=0) - padding_tensor
        upper = candidate_positions.amax(dim=0) + padding_tensor
        if not bool(torch.all(upper > lower)):
            raise ValueError(
                "catalog positions cannot define a positive-area bbox; specify "
                "bbox/polygon or use positive automatic padding"
            )
        border = _bbox_border(lower[0], lower[1], upper[0], upper[1])
        spatial_kind = "bbox"
        spatial_inferred = True

    metadata = {
        "time_unit": catalog.metadata.get("time_unit", "unspecified"),
        "distance_unit": catalog.metadata.get("distance_unit", "unspecified"),
        "coordinate_kind": catalog.metadata.get("coordinate_kind", "unspecified"),
        "coordinate_system": catalog.metadata.get(
            "coordinate_system", "unspecified"
        ),
        "time_zero_inferred": zero_inferred,
        "time_start_inferred": start_inferred,
        "time_end_inferred": end_inferred,
        "spatial_inferred": spatial_inferred,
        "bbox_inferred_from": (
            "time_filtered_targets" if spatial_inferred else None
        ),
        "bbox_inference_event_count": (
            candidate_count if spatial_inferred else None
        ),
        "automatic_padding": padding_tensor.detach().cpu().tolist(),
        "spatial_boundary": "inclusive",
        "tensor_dtype": str(catalog.dtype),
        "tensor_device": str(catalog.device),
    }
    return ObservationDomain(
        time_zero=resolved_zero.clone(),
        time_start=resolved_start.clone(),
        time_end=resolved_end.clone(),
        border=border,
        spatial_kind=spatial_kind,
        include_start=include_start,
        include_end=include_end,
        metadata=metadata,
    )


def _catalog_scalar(
    value: float | torch.Tensor, catalog: EventCatalog, name: str
) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=catalog.dtype, device=catalog.device)
    if result.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    return result


def _resolve_padding(
    padding: float | Sequence[float] | torch.Tensor, catalog: EventCatalog
) -> torch.Tensor:
    result = torch.as_tensor(padding, dtype=catalog.dtype, device=catalog.device)
    if result.ndim == 0:
        result = result.repeat(2)
    elif result.shape != (2,):
        raise ValueError("padding must be a nonnegative scalar or (x, y) pair")
    if not bool(torch.all(torch.isfinite(result))):
        raise ValueError("padding must be finite")
    if bool(torch.any(result < 0.0)):
        raise ValueError("padding must be nonnegative")
    return result


def _require_planar_catalog(catalog: EventCatalog) -> None:
    coordinate_kind = str(catalog.metadata.get("coordinate_kind", "")).lower()
    coordinate_system = str(
        catalog.metadata.get("coordinate_system", "")
    ).lower()
    distance_unit = str(catalog.metadata.get("distance_unit", "")).lower()
    if (
        coordinate_kind == "geographic"
        or "4326" in coordinate_system
        or "geographic" in coordinate_system
        or distance_unit in {"degree", "degrees", "deg"}
    ):
        raise ValueError(
            "cannot calculate a planar observation area from longitude/latitude; "
            "call project_geographic_catalog(...) first"
        )
    linear_units = {
        "m",
        "meter",
        "meters",
        "metre",
        "metres",
        "km",
        "kilometer",
        "kilometers",
        "kilometre",
        "kilometres",
    }
    planar_system_tokens = (
        "projected",
        "local",
        "utm",
        "lambert",
        "stereographic",
        "mercator",
    )
    explicitly_planar = (
        coordinate_kind == "projected"
        or distance_unit in linear_units
        or any(token in coordinate_system for token in planar_system_tokens)
    )
    if not explicitly_planar:
        raise ValueError(
            "catalog metadata must identify planar coordinates and a linear "
            "distance unit before resolving an observation area"
        )


def _normalized_metadata_value(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return None if normalized in {"", "unspecified", "none"} else normalized


def _bbox_border(
    x_min: torch.Tensor,
    y_min: torch.Tensor,
    x_max: torch.Tensor,
    y_max: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        (
            torch.stack((x_min, y_min)),
            torch.stack((x_max, y_min)),
            torch.stack((x_max, y_max)),
            torch.stack((x_min, y_max)),
        )
    )


def _signed_twice_area(border: torch.Tensor) -> torch.Tensor:
    # Translating the polygon close to the origin avoids cancellation when a
    # small regional domain uses large projected eastings/northings.
    translated = border - border[0]
    following = torch.roll(translated, shifts=-1, dims=0)
    return torch.sum(
        translated[:, 0] * following[:, 1]
        - following[:, 0] * translated[:, 1]
    )


def _normalize_polygon(border: torch.Tensor) -> torch.Tensor:
    if border.ndim != 2 or border.shape[1] != 2:
        raise ValueError("border must have shape [N_vertices, 2]")
    if border.shape[0] >= 2 and bool(torch.equal(border[0], border[-1])):
        border = border[:-1]
    if border.shape[0] < 3:
        raise ValueError("border must contain at least three vertices")
    if torch.unique(border, dim=0).shape[0] != border.shape[0]:
        raise ValueError("border vertices must be distinct")

    signed_area = _signed_twice_area(border)
    extent = border.amax(dim=0) - border.amin(dim=0)
    scale = torch.maximum(
        extent[0] * extent[1],
        torch.ones((), dtype=border.dtype, device=border.device),
    )
    tolerance = torch.finfo(border.dtype).eps * scale * border.shape[0] * 16.0
    if bool(torch.abs(signed_area) <= tolerance):
        raise ValueError("border must define a non-zero planar area")
    if bool(signed_area < 0.0):
        border = torch.flip(border, dims=(0,))

    vertices = border.detach().cpu().tolist()
    start = min(range(len(vertices)), key=lambda index: tuple(vertices[index]))
    if start:
        border = torch.roll(border, shifts=-start, dims=0)
        vertices = border.detach().cpu().tolist()
    extent_value = float(extent.abs().amax().item())
    intersection_tolerance = (
        torch.finfo(border.dtype).eps * max(extent_value * extent_value, 1.0) * 32.0
    )
    if not _is_simple_polygon(vertices, intersection_tolerance):
        raise ValueError("border must be a simple polygon without self-intersections")
    return border


def _is_simple_polygon(
    vertices: list[list[float]], tolerance: float
) -> bool:
    vertex_count = len(vertices)
    for first_index in range(vertex_count):
        first_start = vertices[first_index]
        first_end = vertices[(first_index + 1) % vertex_count]
        for second_index in range(first_index + 1, vertex_count):
            if second_index in {
                first_index,
                (first_index + 1) % vertex_count,
                (first_index - 1) % vertex_count,
            }:
                continue
            second_start = vertices[second_index]
            second_end = vertices[(second_index + 1) % vertex_count]
            if _segments_intersect(
                first_start,
                first_end,
                second_start,
                second_end,
                tolerance,
            ):
                return False
    return True


def _segments_intersect(
    first_start: list[float],
    first_end: list[float],
    second_start: list[float],
    second_end: list[float],
    tolerance: float,
) -> bool:
    def orientation(a: list[float], b: list[float], c: list[float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (
            c[0] - a[0]
        )

    def on_segment(a: list[float], b: list[float], c: list[float]) -> bool:
        return (
            min(a[0], b[0]) - tolerance <= c[0] <= max(a[0], b[0]) + tolerance
            and min(a[1], b[1]) - tolerance
            <= c[1]
            <= max(a[1], b[1]) + tolerance
        )

    values = (
        orientation(first_start, first_end, second_start),
        orientation(first_start, first_end, second_end),
        orientation(second_start, second_end, first_start),
        orientation(second_start, second_end, first_end),
    )
    signs = [
        0 if abs(value) <= tolerance else (1 if value > 0 else -1)
        for value in values
    ]
    if signs[0] * signs[1] < 0 and signs[2] * signs[3] < 0:
        return True
    return (
        (signs[0] == 0 and on_segment(first_start, first_end, second_start))
        or (signs[1] == 0 and on_segment(first_start, first_end, second_end))
        or (signs[2] == 0 and on_segment(second_start, second_end, first_start))
        or (signs[3] == 0 and on_segment(second_start, second_end, first_end))
    )


def _polygon_covers_points(
    border: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    x = points[:, 0]
    y = points[:, 1]
    inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    boundary = torch.zeros_like(inside)
    epsilon = torch.finfo(points.dtype).eps

    for index in range(border.shape[0]):
        start = border[index]
        end = border[(index + 1) % border.shape[0]]
        edge_x = end[0] - start[0]
        edge_y = end[1] - start[1]
        relative_x = x - start[0]
        relative_y = y - start[1]
        cross = relative_x * edge_y - relative_y * edge_x
        edge_scale = torch.maximum(
            torch.maximum(torch.abs(edge_x), torch.abs(edge_y)),
            torch.ones((), dtype=points.dtype, device=points.device),
        )
        point_scale = torch.maximum(
            torch.maximum(torch.abs(relative_x), torch.abs(relative_y)),
            torch.ones_like(relative_x),
        )
        cross_tolerance = epsilon * edge_scale * point_scale * 16.0
        dot = relative_x * (x - end[0]) + relative_y * (y - end[1])
        boundary |= (torch.abs(cross) <= cross_tolerance) & (
            dot <= epsilon * point_scale.square() * 16.0
        )

        crosses_vertical_range = (start[1] > y) != (end[1] > y)
        safe_edge_y = torch.where(
            torch.abs(edge_y) > epsilon,
            edge_y,
            torch.ones((), dtype=points.dtype, device=points.device),
        )
        intersection_x = start[0] + edge_x * (y - start[1]) / safe_edge_y
        inside ^= crosses_vertical_range & (x < intersection_x)
    return inside | boundary


__all__ = ["ObservationDomain", "resolve_domain"]

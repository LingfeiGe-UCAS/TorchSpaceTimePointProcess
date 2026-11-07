"""Reusable forward and inverse coordinate transforms."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .catalog import EventCatalog


_ProjectionMethod = Literal["legacy_local_approx", "local_equirectangular", "crs"]


@dataclass(frozen=True)
class ProjectTransformer:
    """A reproducible coordinate transform shared by catalogs and domains.

    ``local_equirectangular`` is implemented directly with PyTorch and returns
    local east/north coordinates in kilometres.  ``crs`` delegates to PyProj
    and supports any explicit source/target CRS pair.  The same immutable
    instance performs both forward and inverse transformations, preventing
    catalogs and domain borders from silently using different origins or CRSs.
    """

    method: _ProjectionMethod
    source_crs: str
    target_crs: str
    target_distance_unit: str
    origin_longitude: float | None = None
    origin_latitude: float | None = None
    earth_radius_km: float = 6371.0088

    def __post_init__(self) -> None:
        if self.method not in {
            "legacy_local_approx", "local_equirectangular", "crs"
        }:
            raise ValueError(
                "method must be 'legacy_local_approx', "
                "'local_equirectangular', or 'crs'"
            )
        if not self.source_crs or not self.target_crs:
            raise ValueError("source_crs and target_crs must be non-empty")
        if not self.target_distance_unit:
            raise ValueError("target_distance_unit must be non-empty")
        if self.method in {"legacy_local_approx", "local_equirectangular"}:
            if self.origin_latitude is None:
                raise ValueError(f"{self.method} requires a reference latitude")
            if not math.isfinite(self.origin_latitude) or not (
                -90.0 < self.origin_latitude < 90.0
            ):
                raise ValueError("origin_latitude must lie strictly inside (-90, 90)")
        if self.method == "local_equirectangular":
            if self.origin_longitude is None or self.origin_latitude is None:
                raise ValueError(
                    "local_equirectangular requires longitude/latitude origin"
                )
            if not math.isfinite(self.origin_longitude) or not (
                -180.0 <= self.origin_longitude <= 180.0
            ):
                raise ValueError("origin_longitude must lie in [-180, 180]")
            if not math.isfinite(self.earth_radius_km) or self.earth_radius_km <= 0:
                raise ValueError("earth_radius_km must be finite and positive")

    @classmethod
    def legacy_local_approx(
        cls,
        *,
        reference_latitude: float,
        source_crs: str = "EPSG:4326",
    ) -> "ProjectTransformer":
        """Reproduce the old ETAS ``LocalApprox`` coordinate transform.

        The legacy code used ``x = longitude * cos(reference_latitude)`` and
        ``y = latitude`` without subtracting an origin or converting degrees
        to kilometres.  Keeping this method explicit prevents its scaled-degree
        units from being confused with :meth:`local_equirectangular`.
        """

        return cls(
            method="legacy_local_approx",
            source_crs=source_crs,
            target_crs="legacy_local_approx",
            target_distance_unit="scaled_degree",
            origin_latitude=float(reference_latitude),
        )

    @classmethod
    def local_equirectangular(
        cls,
        *,
        origin_longitude: float,
        origin_latitude: float,
        earth_radius_km: float = 6371.0088,
        source_crs: str = "EPSG:4326",
    ) -> "ProjectTransformer":
        return cls(
            method="local_equirectangular",
            source_crs=source_crs,
            target_crs="local_equirectangular",
            target_distance_unit="km",
            origin_longitude=float(origin_longitude),
            origin_latitude=float(origin_latitude),
            earth_radius_km=float(earth_radius_km),
        )

    @classmethod
    def fit_local(
        cls,
        catalog: EventCatalog,
        *,
        earth_radius_km: float = 6371.0088,
    ) -> "ProjectTransformer":
        """Fit one fixed local origin from a geographic reference catalog."""

        _require_geographic_catalog(catalog)
        return cls.local_equirectangular(
            origin_longitude=float(catalog.position[:, 0].mean().item()),
            origin_latitude=float(catalog.position[:, 1].mean().item()),
            earth_radius_km=earth_radius_km,
            source_crs=str(catalog.metadata.get("coordinate_system", "EPSG:4326")),
        )

    @classmethod
    def from_crs(
        cls,
        source_crs: str,
        target_crs: str,
        *,
        target_distance_unit: str = "m",
    ) -> "ProjectTransformer":
        """Create an arbitrary reversible CRS transform backed by PyProj."""

        return cls(
            method="crs",
            source_crs=source_crs,
            target_crs=target_crs,
            target_distance_unit=target_distance_unit,
        )

    @staticmethod
    def _validate_points(points: torch.Tensor) -> None:
        if not isinstance(points, torch.Tensor):
            raise TypeError("points must be a torch.Tensor")
        if not points.is_floating_point():
            raise TypeError("points must use a floating-point dtype")
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape [N, 2]")
        if not bool(torch.all(torch.isfinite(points))):
            raise ValueError("points contain NaN or infinite values")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Transform source-coordinate points to the projected coordinates."""

        self._validate_points(points)
        if self.method in {"legacy_local_approx", "local_equirectangular"}:
            longitude = points[:, 0]
            latitude = points[:, 1]
            if bool(torch.any(longitude < -180.0)) or bool(
                torch.any(longitude > 180.0)
            ):
                raise ValueError("longitude values must lie in [-180, 180]")
            if bool(torch.any(latitude < -90.0)) or bool(
                torch.any(latitude > 90.0)
            ):
                raise ValueError("latitude values must lie in [-90, 90]")
            radians = math.pi / 180.0
            if self.method == "legacy_local_approx":
                x = longitude * math.cos(float(self.origin_latitude) * radians)
                return torch.stack((x, latitude), dim=1)
            x = (
                self.earth_radius_km
                * math.cos(float(self.origin_latitude) * radians)
                * (longitude - float(self.origin_longitude))
                * radians
            )
            y = (
                self.earth_radius_km
                * (latitude - float(self.origin_latitude))
                * radians
            )
            return torch.stack((x, y), dim=1)
        return self._pyproj_transform(points, inverse=False)

    def inverse(self, points: torch.Tensor) -> torch.Tensor:
        """Transform projected points back to the source coordinates."""

        self._validate_points(points)
        if self.method in {"legacy_local_approx", "local_equirectangular"}:
            radians = math.pi / 180.0
            if self.method == "legacy_local_approx":
                longitude = points[:, 0] / math.cos(
                    float(self.origin_latitude) * radians
                )
                return torch.stack((longitude, points[:, 1]), dim=1)
            longitude = points[:, 0] / (
                self.earth_radius_km
                * math.cos(float(self.origin_latitude) * radians)
                * radians
            ) + float(self.origin_longitude)
            latitude = (
                points[:, 1] / (self.earth_radius_km * radians)
                + float(self.origin_latitude)
            )
            return torch.stack((longitude, latitude), dim=1)
        return self._pyproj_transform(points, inverse=True)

    def _pyproj_transform(
        self, points: torch.Tensor, *, inverse: bool
    ) -> torch.Tensor:
        try:
            from pyproj import Transformer
        except ImportError as error:
            raise ImportError(
                "arbitrary CRS transforms require pyproj>=3.6"
            ) from error
        source = self.target_crs if inverse else self.source_crs
        target = self.source_crs if inverse else self.target_crs
        transformer = Transformer.from_crs(source, target, always_xy=True)
        values = points.detach().cpu().numpy()
        x, y = transformer.transform(values[:, 0], values[:, 1])
        return torch.stack(
            (
                torch.as_tensor(x, dtype=points.dtype, device=points.device),
                torch.as_tensor(y, dtype=points.dtype, device=points.device),
            ),
            dim=1,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "source_crs": self.source_crs,
            "target_crs": self.target_crs,
            "target_distance_unit": self.target_distance_unit,
            "origin_longitude_deg": self.origin_longitude,
            "origin_latitude_deg": self.origin_latitude,
            "reference_latitude_deg": (
                self.origin_latitude
                if self.method == "legacy_local_approx"
                else None
            ),
            "earth_radius_km": self.earth_radius_km,
            "axis_order": "x_east_y_north",
        }

    def forward_catalog(self, catalog: EventCatalog) -> EventCatalog:
        if self.method in {"legacy_local_approx", "local_equirectangular"}:
            _require_geographic_catalog(catalog)
        else:
            _require_catalog_crs(catalog, self.source_crs)
        metadata = copy.deepcopy(dict(catalog.metadata))
        metadata.update(
            {
                "source_coordinate_kind": metadata.get(
                    "coordinate_kind", "geographic"
                ),
                "source_coordinate_system": metadata.get(
                    "coordinate_system", self.source_crs
                ),
                "source_distance_unit": metadata.get("distance_unit", "degree"),
                "coordinate_kind": "projected",
                "coordinate_system": self.target_crs,
                "distance_unit": self.target_distance_unit,
                "projection": self.metadata(),
            }
        )
        return _catalog_with_position(catalog, self.forward(catalog.position), metadata)

    def inverse_catalog(self, catalog: EventCatalog) -> EventCatalog:
        coordinate_kind = str(catalog.metadata.get("coordinate_kind", "")).lower()
        if coordinate_kind != "projected":
            raise ValueError("inverse_catalog requires a projected catalog")
        _require_matching_projection(catalog.metadata, self)
        metadata = copy.deepcopy(dict(catalog.metadata))
        metadata.update(
            {
                "coordinate_kind": metadata.get(
                    "source_coordinate_kind", "geographic"
                ),
                "coordinate_system": metadata.get(
                    "source_coordinate_system", self.source_crs
                ),
                "distance_unit": metadata.get("source_distance_unit", "degree"),
                "inverse_projection": self.metadata(),
            }
        )
        return _catalog_with_position(catalog, self.inverse(catalog.position), metadata)

    def forward_domain(self, domain: Any) -> Any:
        if not hasattr(domain, "border"):
            raise TypeError("domain must provide border")
        return domain.with_border(
            self.forward(domain.border),
            source_coordinate_kind=domain.metadata.get(
                "coordinate_kind", "unspecified"
            ),
            source_coordinate_system=domain.metadata.get(
                "coordinate_system", self.source_crs
            ),
            source_distance_unit=domain.metadata.get(
                "distance_unit", "unspecified"
            ),
            coordinate_kind="projected",
            coordinate_system=self.target_crs,
            distance_unit=self.target_distance_unit,
            projection=self.metadata(),
        )

    def inverse_domain(self, domain: Any) -> Any:
        if not hasattr(domain, "border"):
            raise TypeError("domain must provide border")
        _require_matching_projection(domain.metadata, self)
        return domain.with_border(
            self.inverse(domain.border),
            coordinate_kind=domain.metadata.get(
                "source_coordinate_kind", "geographic"
            ),
            coordinate_system=domain.metadata.get(
                "source_coordinate_system", self.source_crs
            ),
            distance_unit=domain.metadata.get("source_distance_unit", "degree"),
            inverse_projection=self.metadata(),
        )


def _require_geographic_catalog(catalog: EventCatalog) -> None:
    coordinate_kind = str(catalog.metadata.get("coordinate_kind", "")).lower()
    coordinate_system = str(catalog.metadata.get("coordinate_system", "")).lower()
    distance_unit = str(catalog.metadata.get("distance_unit", "")).lower()
    if not (
        coordinate_kind == "geographic"
        or "4326" in coordinate_system
        or distance_unit in {"degree", "degrees", "deg"}
    ):
        raise ValueError(
            "catalog metadata must identify longitude/latitude coordinates"
        )


def _require_catalog_crs(catalog: EventCatalog, source_crs: str) -> None:
    configured = str(catalog.metadata.get("coordinate_system", "")).strip()
    if not configured or configured.lower() == "unspecified":
        raise ValueError("catalog metadata must identify its source CRS")
    try:
        from pyproj import CRS
    except ImportError as error:
        raise ImportError("CRS validation requires pyproj>=3.6") from error
    if CRS.from_user_input(configured) != CRS.from_user_input(source_crs):
        raise ValueError(
            f"catalog CRS '{configured}' does not match transformer source CRS "
            f"'{source_crs}'"
        )


def _require_matching_projection(
    metadata: Any, transformer: ProjectTransformer
) -> None:
    recorded = metadata.get("projection") if hasattr(metadata, "get") else None
    if recorded is not None and dict(recorded) != transformer.metadata():
        raise ValueError(
            "the supplied transformer does not match the recorded forward projection"
        )


def _catalog_with_position(
    catalog: EventCatalog,
    position: torch.Tensor,
    metadata: dict[str, Any],
) -> EventCatalog:
    return EventCatalog(
        event_id=catalog.event_id,
        sequence_index=catalog.sequence_index,
        time=catalog.time,
        position=position,
        magnitude=catalog.magnitude,
        completeness_magnitude=catalog.completeness_magnitude,
        depth=catalog.depth,
        inside=catalog.inside,
        finite_source_id=catalog.finite_source_id,
        metadata=metadata,
    )


def project_geographic_catalog(
    catalog: EventCatalog,
    *,
    origin_longitude: float | None = None,
    origin_latitude: float | None = None,
    earth_radius_km: float = 6371.0088,
) -> EventCatalog:
    """Backward-compatible convenience wrapper for local projection."""

    if origin_longitude is None and origin_latitude is None:
        transformer = ProjectTransformer.fit_local(
            catalog, earth_radius_km=earth_radius_km
        )
    elif origin_longitude is not None and origin_latitude is not None:
        transformer = ProjectTransformer.local_equirectangular(
            origin_longitude=origin_longitude,
            origin_latitude=origin_latitude,
            earth_radius_km=earth_radius_km,
            source_crs=str(catalog.metadata.get("coordinate_system", "EPSG:4326")),
        )
    else:
        raise ValueError(
            "origin_longitude and origin_latitude must be provided together"
        )
    return transformer.forward_catalog(catalog)

"""Thin builders connecting TOML sections to the existing public classes."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from ..data import (
    EventCatalog,
    ObservationDomain,
    ProjectTransformer,
    knn_bandwidth,
    resolve_domain,
)
from ..finite_source import (
    FiniteSourceGeometry,
    load_legacy_finite_source_geometry,
)
from ..inference import (
    AdaptiveReferenceInference,
    FortranEMLBFGS,
    LegacyEMLBFGS,
    LegacyEMLBFGSConfig,
    TriggerWeightedMagnitudeSelector,
)
from ..integration import InfiniteSpaceIntegrator, PolygonSpatialIntegrator
from ..io import CatalogColumns, load_catalog
from ..likelihood import ExactLikelihood
from ..models.backgrounds import (
    BackgroundModel,
    GaussianBetaKDEBackground,
    GaussianKDEBackground,
    KDEState,
)
from ..models.etas import (
    ETAS2D,
    ETAS2DParameters,
    ETAS3D,
    ETAS3DParameters,
    FiniteSourceETAS3D,
    FiniteSourceETAS3DParameters,
)
from ..models.parameters import ParameterConfig
from ..results import FittedRun, compute_event_state
from .loader import TorchSTPPConfig, load_config


ModelFactory = Callable[[Mapping[str, Any], Any, Any], torch.nn.Module]


@dataclass
class ConfiguredProject:
    """Objects required for one run, built from one reproducible configuration."""

    config: TorchSTPPConfig
    catalog: EventCatalog
    domain: ObservationDomain
    transformer: ProjectTransformer | None
    model: torch.nn.Module
    likelihood: ExactLikelihood
    inference: Any
    background_state: KDEState | None = None

    def fit(self, *, reset_peak_memory: bool = True) -> FittedRun:
        """Run configured inference and retain reusable post-fit event state."""

        if self.background_state is None:
            raise ValueError(
                "background bandwidth is required; pass bandwidth to "
                "build_project(...) or call create_background_state(...)"
            )
        cuda_run = self.catalog.device.type == "cuda"
        if cuda_run and reset_peak_memory:
            torch.cuda.set_device(self.catalog.device)
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        fit_result = self.inference.fit(
            self.model,
            self.likelihood,
            self.catalog,
            self.domain,
            self.background_state,
        )
        elapsed = time.perf_counter() - started
        self.background_state = fit_result.background_state
        event_state = compute_event_state(
            self.model,
            self.catalog,
            self.background_state,
            target_chunk_size=self.likelihood.target_chunk_size,
            source_chunk_size=self.likelihood.source_chunk_size,
        )
        return FittedRun(
            model=self.model,
            catalog=self.catalog,
            domain=self.domain,
            background_state=self.background_state,
            event_state=event_state,
            likelihood=self.likelihood,
            fit_result=fit_result,
            config=dict(self.config.data),
            elapsed_seconds=elapsed,
            fit_elapsed_seconds=elapsed,
            peak_memory_bytes=(
                torch.cuda.max_memory_allocated()
                if cuda_run
                else None
            ),
        )

    def create_background_state(
        self, bandwidth: float | Sequence[float] | torch.Tensor
    ) -> KDEState:
        """Attach the external/adaptive KDE bandwidth without changing Catalog."""

        state_section = self.config.section("background").get("state", {})
        initial_probability = float(state_section.get("initial_probability", 1.0))
        if not 0.0 <= initial_probability <= 1.0:
            raise ValueError("background.state.initial_probability must be in [0, 1]")
        values = torch.as_tensor(
            bandwidth, dtype=self.catalog.dtype, device=self.catalog.device
        )
        if values.ndim == 0:
            values = values.expand(len(self.catalog)).clone()
        if values.shape != self.catalog.time.shape:
            raise ValueError("bandwidth must be a scalar or have shape [N_events]")
        duration_setting = state_section.get("duration", "observation")
        if duration_setting == "observation":
            duration = self.domain.duration
        elif duration_setting == "history":
            duration = self.domain.history_duration
        elif isinstance(duration_setting, (int, float)):
            duration = float(duration_setting)
        else:
            raise ValueError(
                "background.state.duration must be 'observation', 'history', "
                "or a positive number"
            )
        state = KDEState(
            position=self.catalog.position,
            bandwidth=values,
            weights=torch.full_like(self.catalog.time, initial_probability),
            duration=duration,
            depth=self.catalog.depth,
        )
        state.validate()
        self.background_state = state
        return state


def build_project(
    config: str | Path | TorchSTPPConfig,
    *,
    catalog: EventCatalog | None = None,
    bandwidth: float | Sequence[float] | torch.Tensor | None = None,
    model_factories: Mapping[str, ModelFactory] | None = None,
) -> ConfiguredProject:
    """Build the standard run components while keeping existing APIs intact.

    A caller may supply an in-memory catalog and/or custom model factories.
    This makes notebooks concise without forcing custom models to know TOML.
    """

    loaded = load_config(config) if not isinstance(config, TorchSTPPConfig) else config
    runtime = loaded.section("runtime")
    dtype = _dtype(runtime.get("dtype", "float32"))
    device = torch.device(runtime.get("device", "cpu"))
    domain_config = loaded.section("domain")

    if catalog is None:
        catalog = build_catalog(loaded, dtype=dtype, domain_config=domain_config)
    else:
        duplicate_config = loaded.section("catalog").get("duplicates", {})
        catalog = catalog.to(dtype=dtype).resolve_duplicates(
            time_policy=duplicate_config.get("time_policy", "jitter"),
            position_policy=duplicate_config.get("position_policy", "jitter"),
            position_minimum_separation_m=duplicate_config.get(
                "position_minimum_separation_m", 0.05
            ),
            depth_policy=duplicate_config.get("depth_policy", "keep"),
            depth_minimum_separation=duplicate_config.get(
                "depth_minimum_separation", 0.05
            ),
            warn=bool(duplicate_config.get("warn", True)),
        )
        catalog = catalog.filter_for_analysis(
            time_zero=domain_config.get("time_zero"),
            time_end=domain_config.get("time_end"),
        )
        catalog = _filter_catalog_depth(
            catalog, loaded.section("catalog").get("depth_filter")
        )

    raw_polygon = _domain_polygon(loaded, domain_config)
    raw_bbox = domain_config.get("bbox")
    transformer = build_transformer(
        loaded, catalog, polygon=raw_polygon, bbox=raw_bbox
    )
    finite_source_geometry = build_finite_source_geometry(
        loaded,
        dtype=dtype,
    )
    if transformer is not None:
        catalog = transformer.forward_catalog(catalog)
        if finite_source_geometry is not None:
            finite_source_geometry = finite_source_geometry.with_position(
                transformer.forward(finite_source_geometry.position),
                coordinate_kind="projected",
                coordinate_system=transformer.target_crs,
                distance_unit=transformer.target_distance_unit,
                projection=transformer.metadata(),
            )
        raw_polygon = _forward_vertices(transformer, raw_polygon, catalog)
        raw_bbox = _forward_bbox(transformer, raw_bbox, catalog)

    domain = resolve_domain(
        catalog,
        time_zero=domain_config.get("time_zero"),
        time_start=domain_config.get("time_start"),
        time_end=domain_config.get("time_end"),
        polygon=raw_polygon,
        bbox=raw_bbox,
        padding=domain_config.get("padding", 0.0),
        include_start=bool(domain_config.get("include_start", True)),
        include_end=bool(domain_config.get("include_end", True)),
    )

    background = build_background(loaded)
    model = build_model(
        loaded,
        background=background,
        finite_source_geometry=finite_source_geometry,
        model_factories=model_factories,
        coordinate_distance_unit=str(
            catalog.metadata.get("distance_unit", "unspecified")
        ),
    )
    _validate_catalog_for_model(catalog, model)
    likelihood = build_likelihood(loaded)
    inference = build_inference(loaded, likelihood=likelihood)

    catalog = catalog.to(device=device, dtype=dtype)
    domain = domain.to(device=device, dtype=dtype)
    model = model.to(device=device, dtype=dtype)
    project = ConfiguredProject(
        config=loaded,
        catalog=catalog,
        domain=domain,
        transformer=transformer,
        model=model,
        likelihood=likelihood,
        inference=inference,
    )
    resolved_bandwidth = build_bandwidth(loaded, catalog, external=bandwidth)
    if resolved_bandwidth is not None:
        project.create_background_state(resolved_bandwidth)
    return project


def build_bandwidth(
    config: TorchSTPPConfig,
    catalog: EventCatalog,
    *,
    external: float | Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor | float | Sequence[float] | None:
    """Resolve KDE bandwidth from external data, a constant, or legacy k-NN."""

    if external is not None:
        return external
    background = config.section("background")
    section = background.get("bandwidth", {})
    legacy_source = background.get("state", {}).get("bandwidth_source")
    method = section.get("method", legacy_source or "knn")
    if method == "external":
        return None
    if method == "constant":
        if "value" not in section:
            raise ValueError("constant bandwidth requires background.bandwidth.value")
        return float(section["value"])
    if method == "knn":
        return knn_bandwidth(
            catalog.position,
            neighbors=int(section.get("neighbors", 5)),
            minimum=float(section.get("minimum", 0.05)),
            target_chunk_size=int(section.get("target_chunk_size", 2048)),
            source_chunk_size=int(section.get("source_chunk_size", 4096)),
        )
    raise ValueError(f"unknown background bandwidth method: {method!r}")


def build_catalog(
    config: TorchSTPPConfig,
    *,
    dtype: torch.dtype,
    domain_config: Mapping[str, Any] | None = None,
) -> EventCatalog:
    section = config.section("catalog", required=True)
    if "path" not in section:
        raise ValueError("[catalog].path is required when catalog is not supplied")
    columns = section.get("columns")
    preset = section.get("columns_preset")
    if columns is not None and preset is not None:
        raise ValueError("use either catalog.columns or columns_preset, not both")
    if preset == "geographic_earthquake":
        column_spec: CatalogColumns | Mapping[str, str | None] | None = (
            CatalogColumns.geographic_earthquake(
                event_id=section.get("event_id_column")
            )
        )
    elif preset == "legacy_etas":
        column_spec = CatalogColumns.legacy_etas()
    elif preset in {None, "default"}:
        column_spec = columns
    else:
        raise ValueError(f"unknown catalog.columns_preset: {preset!r}")
    domain_section = dict(domain_config or {})
    ignored = {
        "path",
        "columns",
        "columns_preset",
        "event_id_column",
        "duplicates",
        "depth_filter",
    }
    options = {key: value for key, value in section.items() if key not in ignored}
    duplicates = section.get("duplicates", {})
    options.setdefault(
        "time_duplicate_policy", duplicates.get("time_policy", "jitter")
    )
    options.setdefault(
        "position_duplicate_policy", duplicates.get("position_policy", "jitter")
    )
    options.setdefault(
        "position_minimum_separation_m",
        duplicates.get("position_minimum_separation_m", 0.05),
    )
    options.setdefault(
        "depth_duplicate_policy", duplicates.get("depth_policy", "keep")
    )
    options.setdefault(
        "depth_minimum_separation",
        duplicates.get("depth_minimum_separation", 0.05),
    )
    options.setdefault(
        "warn_on_duplicate_resolution", duplicates.get("warn", True)
    )
    if "completeness_magnitude" not in options:
        raise ValueError("[catalog].completeness_magnitude is required")
    options.setdefault("time_zero", domain_section.get("time_zero"))
    options.setdefault("time_end", domain_section.get("time_end"))
    catalog = load_catalog(
        config.resolve_path(section["path"]),
        columns=column_spec,
        dtype=dtype,
        device="cpu",
        **options,
    )
    return _filter_catalog_depth(catalog, section.get("depth_filter"))


def build_transformer(
    config: TorchSTPPConfig,
    catalog: EventCatalog,
    *,
    polygon: Sequence[Sequence[float]] | torch.Tensor | None = None,
    bbox: Sequence[float] | torch.Tensor | None = None,
) -> ProjectTransformer | None:
    section = config.section("projection")
    method = section.get("method", "legacy_local_approx")
    if method in {"none", None}:
        return None
    if method == "legacy_local_approx":
        latitude = section.get("reference_latitude")
        if latitude is None:
            latitude = _reference_latitude(polygon, bbox, catalog)
        return ProjectTransformer.legacy_local_approx(
            reference_latitude=float(latitude),
            source_crs=str(section.get("source_crs", "EPSG:4326")),
        )
    if method == "local_equirectangular":
        longitude = section.get("origin_longitude")
        latitude = section.get("origin_latitude")
        if longitude is None or latitude is None:
            raise ValueError(
                "local_equirectangular requires projection.origin_longitude "
                "and origin_latitude"
            )
        return ProjectTransformer.local_equirectangular(
            origin_longitude=float(longitude),
            origin_latitude=float(latitude),
            earth_radius_km=float(section.get("earth_radius_km", 6371.0088)),
            source_crs=str(section.get("source_crs", "EPSG:4326")),
        )
    if method == "crs":
        if "source_crs" not in section or "target_crs" not in section:
            raise ValueError("crs projection requires source_crs and target_crs")
        return ProjectTransformer.from_crs(
            str(section["source_crs"]),
            str(section["target_crs"]),
            target_distance_unit=str(section.get("target_distance_unit", "m")),
        )
    raise ValueError(f"unknown projection method: {method!r}")


def build_background(config: TorchSTPPConfig) -> BackgroundModel:
    section = config.section("background")
    model_type = str(config.section("model").get("type", "etas2d")).lower()
    default_type = (
        "gaussian_beta_kde"
        if model_type in {"etas3d", "finite_source_etas3d"}
        else "gaussian_kde"
    )
    background_type = str(section.get("type", default_type)).lower()
    integration = section.get("integration", {})
    integrator = _build_integrator(integration, default_source_chunk=4096)
    common = {
        "target_chunk_size": int(section.get("target_chunk_size", 4096)),
        "source_chunk_size": int(section.get("source_chunk_size", 4096)),
        "spatial_integrator": integrator,
    }
    if background_type == "gaussian_kde":
        return GaussianKDEBackground(**common)
    if background_type == "gaussian_beta_kde":
        maximum_depth, boundary_offset = _etas3d_depth_settings(config)
        return GaussianBetaKDEBackground(
            maximum_depth=maximum_depth,
            depth_concentration=float(section.get("depth_concentration", 6.0)),
            boundary_offset=boundary_offset,
            **common,
        )
    raise ValueError(f"unknown background type: {background_type!r}")


def build_model(
    config: TorchSTPPConfig,
    *,
    background: Any,
    finite_source_geometry: FiniteSourceGeometry | None = None,
    model_factories: Mapping[str, ModelFactory] | None = None,
    coordinate_distance_unit: str | None = None,
) -> torch.nn.Module:
    section = config.section("model")
    model_type = str(section.get("type", "etas2d")).lower()
    factories = dict(model_factories or {})
    if model_type in factories:
        return factories[model_type](section, background, config)
    parameter_types = {
        "etas2d": ETAS2DParameters,
        "etas3d": ETAS3DParameters,
        "finite_source_etas3d": FiniteSourceETAS3DParameters,
    }
    if model_type not in parameter_types:
        raise ValueError(
            f"unknown model type {model_type!r}; pass it through model_factories"
        )
    parameter_type = parameter_types[model_type]
    defaults = parameter_type.defaults()
    parameter_section = section.get("parameters", {})
    values = {
        name: _parameter_config(parameter_section.get(name), getattr(defaults, name))
        for name in defaults.__dataclass_fields__
    }
    integration = section.get("integration", {})
    integrator = _build_integrator(integration, default_source_chunk=4096)
    parameters = parameter_type(**values)
    minimum_spatial_scale = _minimum_spatial_scale(
        section,
        coordinate_distance_unit=coordinate_distance_unit,
    )
    if model_type == "etas2d":
        return ETAS2D(
            parameters,
            background=background,
            spatial_integrator=integrator,
            minimum_spatial_scale=minimum_spatial_scale,
        )
    maximum_depth, boundary_offset = _etas3d_depth_settings(config)
    common_3d = {
        "maximum_depth": maximum_depth,
        "depth_boundary_offset": boundary_offset,
        "depth_reference_magnitude_offset": float(
            section.get("depth_reference_magnitude_offset", 0.0)
        ),
        "background": background,
        "spatial_integrator": integrator,
        "minimum_spatial_scale": minimum_spatial_scale,
    }
    if model_type == "etas3d":
        return ETAS3D(parameters, **common_3d)
    if finite_source_geometry is None:
        raise ValueError(
            "model.type='finite_source_etas3d' requires a [finite_source] table"
        )
    finite = config.section("finite_source", required=True)
    return FiniteSourceETAS3D(
        parameters,
        finite_source_geometry=finite_source_geometry,
        patch_chunk_size=int(finite.get("patch_chunk_size", 4096)),
        checkpoint_patches=bool(finite.get("checkpoint_patches", False)),
        horizontal_smoothing_bandwidth=float(
            finite.get("horizontal_smoothing_bandwidth", 0.015)
        ),
        depth_smoothing_concentration=float(
            finite.get("depth_smoothing_concentration", 100.0)
        ),
        tau_normalization=str(finite.get("tau_normalization", "fortran")),
        smoothing_method=str(finite.get("smoothing_method", "blocked")),
        zero_responsibility_policy=str(
            finite.get("zero_responsibility_policy", "retain")
        ),
        **common_3d,
    )


def _minimum_spatial_scale(
    model_section: Mapping[str, Any],
    *,
    coordinate_distance_unit: str | None,
) -> float:
    """Convert an optional physical distance floor to squared model units."""

    resolution = model_section.get("spatial_resolution", {})
    if not isinstance(resolution, Mapping):
        raise TypeError("model.spatial_resolution must be a TOML table")
    if "minimum_distance_m" not in resolution:
        return 0.0
    distance_m = float(resolution["minimum_distance_m"])
    if not math.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError(
            "model.spatial_resolution.minimum_distance_m must be finite "
            "and non-negative"
        )
    if distance_m == 0.0:
        return 0.0

    unit = str(coordinate_distance_unit or "unspecified").strip().lower()
    if unit in {"m", "metre", "metres", "meter", "meters"}:
        coordinate_distance = distance_m
    elif unit in {"km", "kilometre", "kilometres", "kilometer", "kilometers"}:
        coordinate_distance = distance_m / 1000.0
    elif unit == "scaled_degree":
        coordinate_distance = distance_m / 111_320.0
    else:
        raise ValueError(
            "minimum_distance_m requires projected catalog distance units of "
            "m, km, or scaled_degree; got "
            f"{coordinate_distance_unit!r}"
        )
    return coordinate_distance * coordinate_distance


def build_finite_source_geometry(
    config: TorchSTPPConfig,
    *,
    dtype: torch.dtype,
) -> FiniteSourceGeometry | None:
    """Load legacy finite-source patches when the selected model needs them."""

    model_type = str(config.section("model").get("type", "etas2d")).lower()
    finite = config.section("finite_source")
    if model_type != "finite_source_etas3d":
        if finite:
            raise ValueError(
                "[finite_source] is only valid with "
                "model.type='finite_source_etas3d'"
            )
        return None
    if "path" not in finite:
        raise ValueError("[finite_source].path is required")
    expected_columns = [
        "patch_id",
        "longitude",
        "latitude",
        "depth",
        "finite_source_id",
    ]
    columns = finite.get("column_names")
    if columns is not None and list(columns) != expected_columns:
        raise ValueError(
            "finite_source.column_names must describe the legacy five-column "
            f"layout {expected_columns!r}"
        )
    catalog = config.section("catalog")
    return load_legacy_finite_source_geometry(
        config.resolve_path(finite["path"]),
        dtype=dtype,
        device="cpu",
        distance_unit=str(catalog.get("distance_unit", "degree")),
        coordinate_system=str(
            catalog.get("coordinate_system", "EPSG:4326")
        ),
        depth_unit=str(catalog.get("depth_unit", "km")),
        depth_positive=str(catalog.get("depth_positive", "down")),
    )


def build_likelihood(config: TorchSTPPConfig) -> ExactLikelihood:
    section = config.section("likelihood")
    if section.get("type", "exact") != "exact":
        raise ValueError("the standard builder currently supports exact likelihood")
    return ExactLikelihood(
        target_chunk_size=int(section.get("target_chunk_size", 2048)),
        source_chunk_size=int(section.get("source_chunk_size", 4096)),
        checkpoint_blocks=bool(section.get("checkpoint_blocks", False)),
    )


def build_inference(
    config: TorchSTPPConfig, *, likelihood: ExactLikelihood
) -> Any:
    section = config.section("inference")
    inference_type = str(section.get("type", "legacy_em_lbfgs")).lower()
    if inference_type not in {"legacy_em_lbfgs", "fortran_em_lbfgs"}:
        raise ValueError(
            "the standard builder supports legacy_em_lbfgs and fortran_em_lbfgs"
        )
    em = section.get("em", {})
    lbfgs = section.get("lbfgs", {})
    convergence = section.get("convergence", {})
    monitor = section.get("monitor", {})
    depth_reference_auto = section.get("depth_reference_auto", {})
    line_search = lbfgs.get("line_search", "strong_wolfe")
    if line_search == "none":
        line_search = None
    raw_max_iter_schedule = lbfgs.get("max_iter_schedule")
    if raw_max_iter_schedule is not None and not isinstance(
        raw_max_iter_schedule, list
    ):
        raise TypeError("inference.lbfgs.max_iter_schedule must be a TOML array")
    raw_max_eval_schedule = lbfgs.get("max_eval_schedule")
    if raw_max_eval_schedule is not None and not isinstance(
        raw_max_eval_schedule, list
    ):
        raise TypeError("inference.lbfgs.max_eval_schedule must be a TOML array")
    new_retain_history = lbfgs.get("retain_history_across_outer_iterations")
    old_retain_history = lbfgs.get("retain_history_across_e_steps")
    if (
        new_retain_history is not None
        and old_retain_history is not None
        and bool(new_retain_history) != bool(old_retain_history)
    ):
        raise ValueError(
            "retain_history_across_outer_iterations and the deprecated "
            "retain_history_across_e_steps setting disagree"
        )
    retain_history = bool(
        new_retain_history
        if new_retain_history is not None
        else old_retain_history
        if old_retain_history is not None
        else False
    )
    fitter_type = (
        FortranEMLBFGS
        if inference_type == "fortran_em_lbfgs"
        else LegacyEMLBFGS
    )
    base_config = LegacyEMLBFGSConfig(
        outer_iterations=int(em.get("outer_iterations", 11)),
        e_steps_per_outer=int(em.get("e_steps_per_outer", 10)),
        decrease_e_steps=bool(em.get("decrease_e_steps", True)),
        minimum_e_steps=int(em.get("minimum_e_steps", 1)),
        background_tolerance=float(
            convergence.get("background_tolerance", 1e-5)
        ),
        parameter_tolerance=float(
            convergence.get("parameter_tolerance", 1e-5)
        ),
        lbfgs_lr=float(lbfgs.get("lr", 0.05)),
        lbfgs_max_iter=int(lbfgs.get("max_iter", 20)),
        lbfgs_max_iter_schedule=(
            None
            if raw_max_iter_schedule is None
            else tuple(int(value) for value in raw_max_iter_schedule)
        ),
        lbfgs_max_eval=(
            None if lbfgs.get("max_eval") is None else int(lbfgs["max_eval"])
        ),
        lbfgs_max_eval_schedule=(
            None
            if raw_max_eval_schedule is None
            else tuple(int(value) for value in raw_max_eval_schedule)
        ),
        lbfgs_history_size=int(lbfgs.get("history_size", 10)),
        retain_lbfgs_history_across_outer_iterations=retain_history,
        retain_history_across_e_steps=(
            None if old_retain_history is None else bool(old_retain_history)
        ),
        line_search_fn=line_search,
        target_chunk_size=likelihood.target_chunk_size,
        source_chunk_size=likelihood.source_chunk_size,
        print_parameters=bool(monitor.get("print_parameters", False)),
        print_every=int(monitor.get("print_every", 1)),
        stop_on_degenerate=bool(monitor.get("stop_on_degenerate", False)),
        boundary_tolerance=float(monitor.get("boundary_tolerance", 1e-10)),
        maximum_absolute_value=float(monitor.get("maximum_absolute_value", 1e6)),
    )
    base = fitter_type(base_config)
    if not bool(depth_reference_auto.get("enabled", False)):
        return base
    if fitter_type is not FortranEMLBFGS:
        raise ValueError(
            "inference.depth_reference_auto currently requires fortran_em_lbfgs"
        )
    pilot_outer_iterations = int(
        depth_reference_auto.get("pilot_outer_iterations", 3)
    )
    pilot = fitter_type(
        replace(base_config, outer_iterations=pilot_outer_iterations)
    )
    return AdaptiveReferenceInference(
        pilot=pilot,
        base=base,
        selector=TriggerWeightedMagnitudeSelector(
            minimum_offset=float(depth_reference_auto.get("minimum_offset", 0.3)),
            maximum_offset=float(depth_reference_auto.get("maximum_offset", 1.5)),
            round_to=float(depth_reference_auto.get("round_to", 0.1)),
        ),
        pilot_outer_iterations=pilot_outer_iterations,
    )


def _parameter_config(value: Any, default: ParameterConfig) -> ParameterConfig:
    if value is None:
        return default
    if not isinstance(value, Mapping):
        raise TypeError("each model parameter must be a TOML table")
    allowed = {"value", "trainable", "lower_bound", "upper_bound", "transform"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown parameter settings: {sorted(unknown)}")
    return ParameterConfig(
        value=float(value.get("value", default.value)),
        trainable=bool(value.get("trainable", default.trainable)),
        lower_bound=value.get("lower_bound", default.lower_bound),
        upper_bound=value.get("upper_bound", default.upper_bound),
        transform=value.get("transform", default.transform),
    )


def _etas3d_depth_settings(config: TorchSTPPConfig) -> tuple[float, float]:
    section = config.section("model")
    if "maximum_depth" not in section:
        raise ValueError("[model].maximum_depth is required for ETAS3D")
    maximum_depth = float(section["maximum_depth"])
    boundary_offset = float(section.get("depth_boundary_offset", 0.5))
    if maximum_depth <= 0.0:
        raise ValueError("model.maximum_depth must be positive")
    if boundary_offset <= 0.0 or 2.0 * boundary_offset >= maximum_depth:
        raise ValueError(
            "model.depth_boundary_offset must be positive and smaller than "
            "maximum_depth / 2"
        )
    return maximum_depth, boundary_offset


def _validate_catalog_for_model(catalog: EventCatalog, model: torch.nn.Module) -> None:
    maximum_depth = getattr(model, "maximum_depth", None)
    if maximum_depth is None:
        return
    if catalog.depth is None:
        raise ValueError("the selected 3-D model requires a catalog depth column")
    if not bool(torch.all(torch.isfinite(catalog.depth))):
        raise ValueError("3-D catalog depth contains NaN or infinite values")
    if bool(torch.any(catalog.depth < 0.0)) or bool(
        torch.any(catalog.depth > maximum_depth)
    ):
        raise ValueError(
            "3-D catalog depth must lie in [0, model.maximum_depth]"
        )
    if str(catalog.metadata.get("depth_positive", "unspecified")).lower() == "up":
        raise ValueError(
            "the selected 3-D model expects positive-down depth; convert the catalog or set "
            "catalog.depth_positive = 'down'"
        )
    if isinstance(model, FiniteSourceETAS3D):
        if catalog.finite_source_id is None:
            raise ValueError(
                "finite_source_etas3d requires catalog.columns.finite_source_id"
            )
        # This validates uniqueness and that every positive catalog ID has a
        # matching patch group before any expensive likelihood work begins.
        model.finite_source_parent_indices(catalog)


def _filter_catalog_depth(
    catalog: EventCatalog, section: Mapping[str, Any] | None
) -> EventCatalog:
    """Apply an explicit depth window while preserving original event IDs."""

    if section is None:
        return catalog
    if not isinstance(section, Mapping):
        raise TypeError("catalog.depth_filter must be a TOML table")
    allowed = {"minimum", "maximum", "include_minimum", "include_maximum"}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"unknown catalog depth filter settings: {sorted(unknown)}")
    if catalog.depth is None:
        raise ValueError("catalog.depth_filter requires a catalog depth column")
    if "minimum" not in section and "maximum" not in section:
        raise ValueError("catalog.depth_filter requires minimum and/or maximum")

    mask = torch.ones_like(catalog.depth, dtype=torch.bool)
    if "minimum" in section:
        minimum = float(section["minimum"])
        if bool(section.get("include_minimum", True)):
            mask &= catalog.depth >= minimum
        else:
            mask &= catalog.depth > minimum
    if "maximum" in section:
        maximum = float(section["maximum"])
        if "minimum" in section and maximum <= float(section["minimum"]):
            raise ValueError("catalog.depth_filter maximum must exceed minimum")
        if bool(section.get("include_maximum", True)):
            mask &= catalog.depth <= maximum
        else:
            mask &= catalog.depth < maximum
    if not bool(torch.any(mask)):
        raise ValueError("catalog.depth_filter removed every event")
    return catalog.take(mask)


def _build_integrator(
    section: Mapping[str, Any], *, default_source_chunk: int
) -> Any:
    kind = section.get("type", "polygon")
    if kind == "polygon":
        return PolygonSpatialIntegrator(
            subdivisions=int(section.get("subdivisions", 1001)),
            source_chunk_size=int(
                section.get("source_chunk_size", default_source_chunk)
            ),
        )
    if kind == "infinite_space":
        return InfiniteSpaceIntegrator()
    raise ValueError(f"unknown spatial integration type: {kind!r}")


def _dtype(value: Any) -> torch.dtype:
    mapping = {"float32": torch.float32, "float64": torch.float64}
    try:
        return mapping[str(value).lower()]
    except KeyError as error:
        raise ValueError("runtime.dtype must be 'float32' or 'float64'") from error


def _domain_polygon(
    config: TorchSTPPConfig, section: Mapping[str, Any]
) -> Sequence[Sequence[float]] | None:
    if "polygon" in section and "polygon_path" in section:
        raise ValueError("use either domain.polygon or polygon_path, not both")
    if "polygon" in section:
        return section["polygon"]
    if "polygon_path" not in section:
        return None
    path = config.resolve_path(section["polygon_path"])
    if not path.is_file():
        raise FileNotFoundError(f"domain polygon does not exist: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        x_name = str(section.get("polygon_x_column", "x"))
        y_name = str(section.get("polygon_y_column", "y"))
        return [[float(row[x_name]), float(row[y_name])] for row in rows]
    raise ValueError("domain.polygon_path must point to .json or .csv")


def _reference_latitude(
    polygon: Sequence[Sequence[float]] | torch.Tensor | None,
    bbox: Sequence[float] | torch.Tensor | None,
    catalog: EventCatalog,
) -> float:
    if polygon is not None:
        points = torch.as_tensor(polygon, dtype=torch.float64)
        x, y = points[:, 0], points[:, 1]
        x2, y2 = torch.roll(x, -1), torch.roll(y, -1)
        cross = x * y2 - x2 * y
        area6 = 3.0 * cross.sum()
        if abs(float(area6)) > torch.finfo(torch.float64).eps:
            return float(((y + y2) * cross).sum() / area6)
        return float(y.mean())
    if bbox is not None:
        bounds = torch.as_tensor(bbox, dtype=torch.float64)
        return float((bounds[1] + bounds[3]) / 2.0)
    return float(catalog.position[:, 1].mean().item())


def _forward_vertices(
    transformer: ProjectTransformer,
    polygon: Sequence[Sequence[float]] | torch.Tensor | None,
    catalog: EventCatalog,
) -> torch.Tensor | None:
    if polygon is None:
        return None
    values = torch.as_tensor(polygon, dtype=catalog.dtype, device=catalog.device)
    return transformer.forward(values)


def _forward_bbox(
    transformer: ProjectTransformer,
    bbox: Sequence[float] | torch.Tensor | None,
    catalog: EventCatalog,
) -> list[float] | None:
    if bbox is None:
        return None
    values = torch.as_tensor(bbox, dtype=catalog.dtype, device=catalog.device)
    corners = torch.stack((values[[0, 1]], values[[2, 1]], values[[2, 3]], values[[0, 3]]))
    projected = transformer.forward(corners)
    lower, upper = projected.amin(dim=0), projected.amax(dim=0)
    return [float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])]

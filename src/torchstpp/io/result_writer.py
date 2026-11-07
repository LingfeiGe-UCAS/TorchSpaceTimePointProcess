import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from ..results import FittedRun


def _cpu_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.numpy()
    return value


def catalog_event_ids(catalog: Any):
    if hasattr(catalog, "event_id"):
        return np.asarray(_cpu_value(catalog.event_id))
    return torch.arange(catalog.time.shape[0], dtype=torch.int64).numpy()


class ResultWriter:
    """Write compact fitted-run products without recomputing the model."""

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def write_summary(self, run: FittedRun) -> Path:
        first_parameter = next(run.model.parameters())
        summary = {
            "negative_log_likelihood": run.fit_result.negative_log_likelihood,
            "converged": run.fit_result.converged,
            "outer_iterations": run.fit_result.outer_iterations,
            "stop_reason": run.fit_result.stop_reason,
            "degenerate": run.fit_result.stop_reason is not None,
            "event_count": int(run.catalog.time.shape[0]),
            "device": str(first_parameter.device),
            "dtype": str(first_parameter.dtype),
            "elapsed_seconds": run.elapsed_seconds,
            "fit_elapsed_seconds": run.fit_elapsed_seconds,
            "peak_memory_bytes": run.peak_memory_bytes,
            "catalog_duplicate_resolution": getattr(
                run.catalog, "metadata", {}
            ).get("duplicate_resolution"),
        }
        if hasattr(run.model, "depth_reference_concentration"):
            summary["depth_reference_parameterization"] = {
                "magnitude_offset": float(
                    run.model.depth_reference_magnitude_offset
                ),
                "concentration": float(
                    run.model.depth_reference_concentration.detach().item()
                ),
                "legacy_eta": float(run.model.eta.detach().item()),
                "delta": float(run.model.delta.detach().item()),
                "selection": getattr(
                    run.model, "depth_reference_selection", None
                ),
            }
        if hasattr(run.model, "minimum_spatial_scale"):
            configured_resolution = (
                run.config.get("model", {}).get("spatial_resolution", {})
            )
            minimum_scale = float(
                run.model.minimum_spatial_scale.detach().item()
            )
            summary["spatial_resolution"] = {
                "minimum_distance_m": float(
                    configured_resolution.get("minimum_distance_m", 0.0)
                ),
                "coordinate_distance_unit": getattr(
                    run.catalog, "metadata", {}
                ).get("distance_unit"),
                "minimum_scale_model_units_squared": minimum_scale,
                "free_d": float(run.model.d.detach().item()),
                "scale_at_completeness": float(
                    (run.model.minimum_spatial_scale + run.model.d)
                    .detach()
                    .item()
                ),
            }
        if hasattr(run.model, "patch_tau"):
            summary["finite_source"] = {
                "patch_count": int(run.model.patch_tau.numel()),
                "group_count": int(
                    run.model.finite_source_group_ids.numel()
                ),
                "group_ids": _cpu_value(
                    run.model.finite_source_group_ids
                ).tolist(),
                "tau_normalization": run.model.tau_normalization,
                "smoothing_method": getattr(run.model, "smoothing_method", "blocked"),
                "zero_responsibility_policy": (
                    run.model.zero_responsibility_policy
                ),
                "patch_chunk_size": run.model.patch_chunk_size,
                "horizontal_smoothing_bandwidth": (
                    run.model.horizontal_smoothing_bandwidth
                ),
                "depth_smoothing_concentration": (
                    run.model.depth_smoothing_concentration
                ),
                "geometry_metadata": dict(
                    getattr(run.model, "finite_source_metadata", {})
                ),
            }
        path = self.output_directory / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path

    def write_parameters(self, run: FittedRun) -> Path:
        configured = run.config.get("model", {}).get("parameters", {})
        path = self.output_directory / "parameters.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            fieldnames = (
                "name", "value", "initial_value", "trainable",
                "lower_bound", "upper_bound", "transform",
            )
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for name, value in run.model.parameter_values().items():
                initial = configured.get(name, {})
                parameter_module = getattr(run.model, f"{name}_parameter", None)
                writer.writerow({
                    "name": name,
                    "value": float(value.detach().item()),
                    "initial_value": initial.get("value", ""),
                    "trainable": initial.get("trainable", True),
                    "lower_bound": initial.get("lower_bound", ""),
                    "upper_bound": initial.get("upper_bound", ""),
                    "transform": getattr(parameter_module, "transform", ""),
                })
        return path

    def write_history(self, run: FittedRun) -> Path:
        names = list(run.model.parameter_values())
        path = self.output_directory / "optimization_history.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "iteration", "negative_log_likelihood",
                "background_change", "parameter_change",
                "m_step_max_iter", "m_step_max_eval", "m_step_iterations",
                "m_step_function_evaluations", *names,
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for step in run.fit_result.history:
                writer.writerow({
                    "iteration": step.iteration,
                    "negative_log_likelihood": step.negative_log_likelihood,
                    "background_change": step.background_change,
                    "parameter_change": step.parameter_change,
                    "m_step_max_iter": step.m_step_max_iter,
                    "m_step_max_eval": step.m_step_max_eval,
                    "m_step_iterations": step.m_step_iterations,
                    "m_step_function_evaluations": (
                        step.m_step_function_evaluations
                    ),
                    **step.parameter_values,
                })
        return path

    def write_events(self, run: FittedRun) -> Path:
        catalog = run.catalog
        data = {
            "event_id": catalog_event_ids(catalog),
            "time": _cpu_value(catalog.time),
            "x": _cpu_value(catalog.position[:, 0]),
            "y": _cpu_value(catalog.position[:, 1]),
            "magnitude": _cpu_value(catalog.magnitude),
            "background_density": _cpu_value(run.event_state.background_density),
            "background_intensity": _cpu_value(run.event_state.background_intensity),
            "total_intensity": _cpu_value(run.event_state.total_intensity),
            "phi": _cpu_value(run.event_state.phi),
        }
        depth = getattr(catalog, "depth", None)
        if depth is not None:
            data["depth"] = _cpu_value(depth)
        finite_source_id = getattr(catalog, "finite_source_id", None)
        if finite_source_id is not None:
            data["finite_source_id"] = _cpu_value(finite_source_id)
        path = self.output_directory / "events.parquet"
        pl.DataFrame(data).write_parquet(path, compression="zstd")
        return path

    def write_subsources(self, run: FittedRun) -> Path | None:
        """Write fitted finite-source patch state, if present."""

        model = run.model
        required = (
            "patch_id",
            "patch_finite_source_id",
            "patch_position",
            "patch_depth",
            "patch_tau",
            "normalized_patch_tau",
            "patch_smoothing_normalizer",
        )
        if not all(hasattr(model, name) for name in required):
            return None
        data = {
            "patch_id": _cpu_value(model.patch_id),
            "finite_source_id": _cpu_value(model.patch_finite_source_id),
            "x": _cpu_value(model.patch_position[:, 0]),
            "y": _cpu_value(model.patch_position[:, 1]),
            "depth": _cpu_value(model.patch_depth),
            "tau": _cpu_value(model.patch_tau),
            "normalized_tau": _cpu_value(model.normalized_patch_tau),
            "smoothing_normalizer": _cpu_value(
                model.patch_smoothing_normalizer
            ),
        }
        path = self.output_directory / "subsources.parquet"
        pl.DataFrame(data).write_parquet(path, compression="zstd")
        return path

    def write_checkpoint(self, run: FittedRun) -> Path:
        def cpu_tensor(value):
            return torch.as_tensor(value).detach().cpu()

        domain_state = {
            "time_zero": _cpu_value(run.domain.time_zero),
            "time_start": _cpu_value(run.domain.time_start),
            "time_end": _cpu_value(run.domain.time_end),
        }
        if hasattr(run.domain, "border"):
            domain_state["border"] = cpu_tensor(run.domain.border)
        checkpoint = {
            "model_state_dict": {
                name: value.detach().cpu()
                for name, value in run.model.state_dict().items()
            },
            "background_state": {
                "position": cpu_tensor(run.background_state.position),
                "bandwidth": cpu_tensor(run.background_state.bandwidth),
                "weights": cpu_tensor(run.background_state.weights),
                "duration": cpu_tensor(run.background_state.duration),
            },
            "event_state": {
                "background_density": cpu_tensor(run.event_state.background_density),
                "background_intensity": cpu_tensor(run.event_state.background_intensity),
                "total_intensity": cpu_tensor(run.event_state.total_intensity),
                "phi": cpu_tensor(run.event_state.phi),
            },
            "domain": domain_state,
            "fit": {
                "negative_log_likelihood": run.fit_result.negative_log_likelihood,
                "converged": run.fit_result.converged,
                "outer_iterations": run.fit_result.outer_iterations,
                "stop_reason": run.fit_result.stop_reason,
            },
            "config": run.config,
            "catalog_metadata": dict(getattr(run.catalog, "metadata", {})),
        }
        if hasattr(run.model, "depth_reference_concentration"):
            checkpoint["model_runtime_metadata"] = {
                "depth_reference_magnitude_offset": float(
                    run.model.depth_reference_magnitude_offset
                ),
                "depth_reference_selection": getattr(
                    run.model, "depth_reference_selection", None
                ),
            }
        if run.background_state.depth is not None:
            checkpoint["background_state"]["depth"] = cpu_tensor(
                run.background_state.depth
            )
        path = self.output_directory / "checkpoint.pt"
        torch.save(checkpoint, path)
        return path

    def write_all(self, run: FittedRun) -> dict[str, Path]:
        paths = {
            "summary": self.write_summary(run),
            "parameters": self.write_parameters(run),
            "history": self.write_history(run),
            "events": self.write_events(run),
            "checkpoint": self.write_checkpoint(run),
        }
        subsources = self.write_subsources(run)
        if subsources is not None:
            paths["subsources"] = subsources
        return paths

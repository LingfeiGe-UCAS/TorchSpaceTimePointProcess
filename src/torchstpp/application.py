"""Complete configuration-driven ETAS application workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from .config import ModelFactory, TorchSTPPConfig, build_project, load_config
from .io import RelationWriter, ResultWriter
from .relations import RelationExtractor
from .results import FittedRun


@dataclass(frozen=True)
class ETASRunResult:
    """In-memory fitted run plus every file produced by the application."""

    run: FittedRun
    output_directory: Path | None
    result_paths: dict[str, Path]
    relation_manifest: dict[str, Any] | None


def run_etas(
    config: str | Path | TorchSTPPConfig,
    *,
    write_outputs: bool | None = None,
    model_factories: Mapping[str, ModelFactory] | None = None,
) -> ETASRunResult:
    """Execute a built-in or externally registered model end to end."""

    total_started = time.perf_counter()
    loaded = load_config(config) if not isinstance(config, TorchSTPPConfig) else config
    model_type = str(loaded.section("model").get("type", "etas2d")).lower()
    if model_type not in {"etas2d", "etas3d"} and (
        model_factories is None or model_type not in model_factories
    ):
        raise ValueError(f"run_etas does not support model.type = {model_type!r}")
    output = loaded.section("output")
    should_write = bool(output.get("enabled", True)) if write_outputs is None else write_outputs
    output_directory: Path | None = None
    if should_write:
        output_directory = loaded.resolve_path(
            output.get("directory", f"../output/{model_type}_run")
        )
        _require_empty_output(output_directory)

    runtime = loaded.section("runtime")
    requested_device = torch.device(runtime.get("device", "cpu"))
    cuda_run = requested_device.type == "cuda"
    if cuda_run:
        torch.cuda.set_device(requested_device)
        torch.cuda.reset_peak_memory_stats()
    project = build_project(loaded, model_factories=model_factories)
    run = project.fit(reset_peak_memory=False)
    paths: dict[str, Path] = {}
    relation_manifest: dict[str, Any] | None = None
    if output_directory is not None:
        writer = ResultWriter(output_directory)
        paths = {
            "parameters": writer.write_parameters(run),
            "history": writer.write_history(run),
            "events": writer.write_events(run),
            "checkpoint": writer.write_checkpoint(run),
        }
        relation = loaded.section("relations")
        if bool(relation.get("enabled", True)):
            extractor = RelationExtractor(
                probability_threshold=float(
                    relation.get("probability_threshold", 1e-4)
                ),
                target_chunk_size=int(
                    relation.get("target_chunk_size", 2048)
                ),
                source_chunk_size=int(
                    relation.get("source_chunk_size", 4096)
                ),
            )
            relation_manifest = RelationWriter(
                output_directory / str(relation.get("directory", "relations"))
            ).write(run, extractor)
        run.elapsed_seconds = time.perf_counter() - total_started
        if cuda_run:
            run.peak_memory_bytes = torch.cuda.max_memory_allocated()
        paths["summary"] = writer.write_summary(run)
    else:
        run.elapsed_seconds = time.perf_counter() - total_started
        if cuda_run:
            run.peak_memory_bytes = torch.cuda.max_memory_allocated()
    return ETASRunResult(
        run=run,
        output_directory=output_directory,
        result_paths=paths,
        relation_manifest=relation_manifest,
    )


# Backward-compatible name retained for notebooks and downstream imports.
ETAS2DRunResult = ETASRunResult


def run_etas2d(
    config: str | Path | TorchSTPPConfig,
    *,
    write_outputs: bool | None = None,
    model_factories: Mapping[str, ModelFactory] | None = None,
) -> ETASRunResult:
    """Run ETAS2D while rejecting an accidentally selected 3-D model."""

    loaded = load_config(config) if not isinstance(config, TorchSTPPConfig) else config
    _require_model_type(loaded, "etas2d", "run_etas2d")
    return run_etas(
        loaded,
        write_outputs=write_outputs,
        model_factories=model_factories,
    )


def run_etas3d(
    config: str | Path | TorchSTPPConfig,
    *,
    write_outputs: bool | None = None,
    model_factories: Mapping[str, ModelFactory] | None = None,
) -> ETASRunResult:
    """Run the point-source ETAS3D workflow from one TOML configuration."""

    loaded = load_config(config) if not isinstance(config, TorchSTPPConfig) else config
    _require_model_type(loaded, "etas3d", "run_etas3d")
    return run_etas(
        loaded,
        write_outputs=write_outputs,
        model_factories=model_factories,
    )


def _require_model_type(
    config: TorchSTPPConfig, expected: str, entry_point: str
) -> None:
    actual = str(config.section("model").get("type", "etas2d")).lower()
    if actual != expected:
        raise ValueError(f"{entry_point} requires model.type = {expected!r}")


def _require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {path}. Choose a new directory "
            "to protect previous scientific results."
        )

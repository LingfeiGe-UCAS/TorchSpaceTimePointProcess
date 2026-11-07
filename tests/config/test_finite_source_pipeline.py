from dataclasses import replace
from pathlib import Path

import polars as pl
import torch

from torchstpp.config import build_project
from torchstpp.inference import FortranEMLBFGS
from torchstpp.io import ResultWriter
from torchstpp.models.etas import FiniteSourceETAS3D


def test_finite_source_config_build_fit_and_outputs(tmp_path: Path):
    catalog_path = tmp_path / "events.etas"
    catalog_path.write_text(
        """1 0.20 0.20 4.0 0.0 10.0 1 1
2 0.30 0.25 3.1 1.0 11.0 0 1
3 0.45 0.35 3.2 2.0 12.0 0 1
4 0.60 0.50 3.3 3.0 13.0 0 1
""",
        encoding="utf-8",
    )
    geometry_path = tmp_path / "sub.etas"
    geometry_path.write_text(
        """1 0.18 0.19 9.5 1
2 0.22 0.21 10.5 1
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "finite.toml"
    config_path.write_text(
        f"""
[runtime]
device = "cpu"
dtype = "float64"

[catalog]
path = "{catalog_path.as_posix()}"
column_names = ["idx", "x", "y", "mag", "time", "depth", "type", "inside"]
completeness_magnitude = 3.0
distance_unit = "km"
depth_unit = "km"
depth_positive = "down"

[catalog.columns]
event_id = "idx"
x = "x"
y = "y"
magnitude = "mag"
time = "time"
depth = "depth"
finite_source_id = "type"
inside = "inside"

[catalog.duplicates]
time_policy = "keep"
position_policy = "keep"
depth_policy = "keep"

[projection]
method = "none"

[domain]
time_zero = 0.0
time_start = 1.0
time_end = 3.0
bbox = [0.0, 0.0, 1.0, 1.0]

[model]
type = "finite_source_etas3d"
maximum_depth = 30.0

[model.integration]
type = "infinite_space"

[finite_source]
path = "{geometry_path.as_posix()}"
column_names = ["patch_id", "longitude", "latitude", "depth", "finite_source_id"]
patch_chunk_size = 1
horizontal_smoothing_bandwidth = 0.1
depth_smoothing_concentration = 10.0
tau_normalization = "corrected"

[background]
type = "gaussian_beta_kde"
depth_concentration = 6.0
target_chunk_size = 2
source_chunk_size = 2

[background.bandwidth]
method = "constant"
value = 0.2

[background.integration]
type = "infinite_space"

[likelihood]
type = "exact"
target_chunk_size = 2
source_chunk_size = 2

[inference]
type = "fortran_em_lbfgs"

[inference.em]
outer_iterations = 1
e_steps_per_outer = 1
decrease_e_steps = false

[inference.lbfgs]
lr = 0.01
max_iter = 1
history_size = 2
line_search = "none"

[inference.convergence]
background_tolerance = 0.0
parameter_tolerance = 0.0
""",
        encoding="utf-8",
    )

    project = build_project(config_path)
    assert isinstance(project.model, FiniteSourceETAS3D)
    assert project.catalog.finite_source_id.tolist() == [1, 0, 0, 0]
    assert project.model.finite_source_group_ids.tolist() == [1]
    assert project.model.patch_count == 2
    assert project.model.tau_normalization == "corrected"

    run = project.fit()
    assert torch.isfinite(
        torch.tensor(run.fit_result.negative_log_likelihood)
    )
    paths = ResultWriter(tmp_path / "output").write_all(run)
    assert "subsources" in paths
    events = pl.read_parquet(paths["events"])
    patches = pl.read_parquet(paths["subsources"])
    assert events["finite_source_id"].to_list() == [1, 0, 0, 0]
    assert patches.height == 2
    assert torch.isclose(
        torch.tensor(
            patches["normalized_tau"].sum(), dtype=torch.float64
        ),
        torch.tensor(1.0, dtype=torch.float64),
    )

    monolithic = build_project(config_path)
    split = build_project(config_path)
    monolithic_fitter = FortranEMLBFGS(
        replace(monolithic.inference.config, outer_iterations=3)
    )
    monolithic_result = monolithic_fitter.fit(
        monolithic.model,
        monolithic.likelihood,
        monolithic.catalog,
        monolithic.domain,
        monolithic.background_state,
    )
    split_state = split.background_state
    split_nll = []
    for _ in range(3):
        split_result = split.inference.fit(
            split.model,
            split.likelihood,
            split.catalog,
            split.domain,
            split_state,
        )
        split_state = split_result.background_state
        split_nll.append(split_result.negative_log_likelihood)

    assert torch.allclose(
        torch.tensor(split_nll, dtype=torch.float64),
        torch.tensor([
            step.negative_log_likelihood for step in monolithic_result.history
        ], dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    for name, monolithic_value in monolithic.model.parameter_values().items():
        assert torch.allclose(
            split.model.parameter_values()[name],
            monolithic_value,
            rtol=1e-12,
            atol=1e-12,
        )
    assert torch.allclose(
        split_state.weights,
        monolithic_result.background_state.weights,
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(
        split.model.normalized_patch_tau,
        monolithic.model.normalized_patch_tau,
        rtol=1e-12,
        atol=1e-12,
    )

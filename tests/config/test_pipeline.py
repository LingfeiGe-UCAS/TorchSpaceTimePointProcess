from pathlib import Path

import torch

from torchstpp.config import build_background, build_project, load_config
from torchstpp.data import EventCatalog
from torchstpp.models.etas import ETAS2D


def test_etas3d_background_depth_concentration_has_default(tmp_path: Path):
    path = tmp_path / "etas3d.toml"
    path.write_text(
        """
[model]
type = "etas3d"
maximum_depth = 40.0
""",
        encoding="utf-8",
    )

    background = build_background(load_config(path))

    assert background.depth_concentration == 6.0


def _catalog() -> EventCatalog:
    dtype = torch.float64
    return EventCatalog(
        event_id=torch.arange(4, dtype=torch.int64),
        time=torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype),
        position=torch.tensor(
            [[100.2, 30.2], [100.4, 30.4], [100.6, 30.6], [100.8, 30.8]],
            dtype=dtype,
        ),
        magnitude=torch.tensor([3.5, 3.6, 3.7, 3.8], dtype=dtype),
        completeness_magnitude=torch.tensor(3.5, dtype=dtype),
        metadata={
            "coordinate_kind": "geographic",
            "coordinate_system": "EPSG:4326",
            "distance_unit": "degree",
            "time_unit": "day",
        },
    )


def test_build_project_connects_sections_without_changing_module_apis(tmp_path: Path):
    path = tmp_path / "run.toml"
    path.write_text(
        """
[runtime]
device = "cpu"
dtype = "float64"

[projection]
method = "legacy_local_approx"

[domain]
time_zero = 0.0
time_start = 1.0
time_end = 3.0
bbox = [100.0, 30.0, 101.0, 31.0]

[model]
type = "etas2d"

[model.spatial_resolution]
minimum_distance_m = 10.0

[model.parameters.alpha]
value = 1.25

[background]
type = "gaussian_kde"
target_chunk_size = 7
source_chunk_size = 8

[background.state]
initial_probability = 0.75
duration = "observation"

[background.integration]
type = "polygon"
subdivisions = 11
source_chunk_size = 9

[likelihood]
type = "exact"
target_chunk_size = 2
source_chunk_size = 3
checkpoint_blocks = true

[inference]
type = "legacy_em_lbfgs"

[inference.em]
outer_iterations = 4
e_steps_per_outer = 2

[inference.lbfgs]
lr = 0.1
max_iter = 5
history_size = 3
line_search = "none"

[inference.convergence]
background_tolerance = 0.001
parameter_tolerance = 0.002
""",
        encoding="utf-8",
    )

    loaded = load_config(path)
    project = build_project(loaded, catalog=_catalog(), bandwidth=0.2)

    assert isinstance(project.model, ETAS2D)
    assert project.transformer is not None
    assert project.transformer.method == "legacy_local_approx"
    assert torch.allclose(project.model.alpha, torch.tensor(1.25, dtype=torch.float64))
    expected_floor = torch.tensor(
        (10.0 / 111_320.0) ** 2, dtype=torch.float64
    )
    assert torch.allclose(project.model.minimum_spatial_scale, expected_floor)
    assert project.likelihood.target_chunk_size == 2
    assert project.inference.config.outer_iterations == 4
    assert project.inference.config.line_search_fn is None
    assert project.background_state is not None
    assert torch.all(project.background_state.weights == 0.75)
    assert float(project.background_state.duration) == 2.0

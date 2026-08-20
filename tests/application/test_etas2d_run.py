import csv
from pathlib import Path

import polars as pl

from torchstpp import run_etas2d


def test_complete_etas2d_file_to_output_run(tmp_path: Path):
    catalog_path = tmp_path / "catalog.csv"
    with catalog_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "longitude", "latitude", "mag"])
        for index in range(6):
            writer.writerow(
                [float(index), 100.1 + 0.12 * index, 30.1 + 0.1 * index, 3.5 + 0.05 * index]
            )
    config_path = tmp_path / "run.toml"
    config_path.write_text(
        f"""
[runtime]
device = "cpu"
dtype = "float64"

[catalog]
path = "{catalog_path.as_posix()}"
columns_preset = "geographic_earthquake"
completeness_magnitude = 3.5
time_unit = "day"
distance_unit = "degree"
coordinate_system = "EPSG:4326"

[projection]
method = "legacy_local_approx"

[domain]
time_zero = 0.0
time_start = 1.0
time_end = 5.0
bbox = [100.0, 30.0, 101.0, 31.0]

[model]
type = "etas2d"

[model.integration]
type = "polygon"
subdivisions = 11
source_chunk_size = 3

[background]
type = "gaussian_kde"
target_chunk_size = 3
source_chunk_size = 3

[background.bandwidth]
method = "knn"
neighbors = 2
minimum = 0.02
target_chunk_size = 2
source_chunk_size = 3

[background.state]
initial_probability = 1.0
duration = "observation"

[background.integration]
type = "polygon"
subdivisions = 11
source_chunk_size = 3

[likelihood]
type = "exact"
target_chunk_size = 2
source_chunk_size = 3
checkpoint_blocks = false

[inference]
type = "legacy_em_lbfgs"

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

[output]
enabled = true
directory = "output"

[relations]
enabled = true
directory = "relations"
probability_threshold = 0.0001
target_chunk_size = 2
source_chunk_size = 3
""",
        encoding="utf-8",
    )

    result = run_etas2d(config_path)

    assert result.run.catalog.time.shape == (6,)
    assert result.run.fit_result.outer_iterations == 1
    assert set(result.result_paths) == {
        "summary", "parameters", "history", "events", "checkpoint"
    }
    assert all(path.is_file() for path in result.result_paths.values())
    events = pl.read_parquet(result.result_paths["events"])
    assert events.height == 6
    assert "phi" in events.columns
    assert result.relation_manifest is not None
    assert (tmp_path / "output" / "relations" / "manifest.json").is_file()

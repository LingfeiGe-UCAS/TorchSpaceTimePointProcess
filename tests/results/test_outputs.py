import json
import tempfile
from pathlib import Path

import polars as pl
import torch

from tests.likelihood.test_exact import make_problem
from torchstpp.inference import LegacyEMLBFGS, LegacyEMLBFGSConfig
from torchstpp.io import RelationWriter, ResultWriter
from torchstpp.likelihood import ExactLikelihood
from torchstpp.relations import RelationExtractor
from torchstpp.results import FittedRun, compute_event_state


def make_fitted_run():
    model, catalog, domain, state = make_problem()
    likelihood = ExactLikelihood(target_chunk_size=2, source_chunk_size=2)
    fitter = LegacyEMLBFGS(LegacyEMLBFGSConfig(
        outer_iterations=1, e_steps_per_outer=1, lbfgs_max_iter=1,
        target_chunk_size=2, source_chunk_size=2,
    ))
    fit_result = fitter.fit(model, likelihood, catalog, domain, state)
    event_state = compute_event_state(
        model, catalog, fit_result.background_state,
        target_chunk_size=2, source_chunk_size=2,
    )
    config = {"model": {"parameters": {
        name: {"value": 1.0, "trainable": True}
        for name in model.parameter_values()
    }}}
    return FittedRun(
        model=model, catalog=catalog, domain=domain,
        background_state=fit_result.background_state,
        event_state=event_state, likelihood=likelihood,
        fit_result=fit_result, config=config,
        elapsed_seconds=1.5, peak_memory_bytes=1024,
    )


def test_event_state_is_reusable_and_normalized():
    run = make_fitted_run()
    event = run.event_state
    assert event.total_intensity.shape == run.catalog.time.shape
    assert torch.allclose(
        event.phi, event.background_intensity / event.total_intensity
    )
    assert torch.all((event.phi >= 0) & (event.phi <= 1))


def test_result_writer_outputs_all_products():
    run = make_fitted_run()
    with tempfile.TemporaryDirectory() as directory:
        paths = ResultWriter(directory).write_all(run)
        assert all(path.exists() for path in paths.values())
        assert json.loads(paths["summary"].read_text())["event_count"] == 6
        events = pl.read_parquet(paths["events"])
        assert "phi" in events.columns and events.height == 6
        checkpoint = torch.load(paths["checkpoint"], weights_only=True)
        assert "model_state_dict" in checkpoint
        assert checkpoint["event_state"]["phi"].shape == (6,)


def test_relations_are_blockwise_filtered_and_written():
    run = make_fitted_run()
    extractor = RelationExtractor(
        probability_threshold=1e-8,
        target_chunk_size=2,
        source_chunk_size=2,
    )
    blocks = list(extractor.iter_relations(run))
    assert blocks
    assert all(torch.all(block.trigger_probability > 1e-8) for block in blocks)
    assert all(torch.all(block.source_index < block.target_index) for block in blocks)

    with tempfile.TemporaryDirectory() as directory:
        manifest = RelationWriter(Path(directory) / "relations").write(run, extractor)
        assert manifest["relation_count"] == sum(block.row_count for block in blocks)
        assert manifest["part_count"] > 0

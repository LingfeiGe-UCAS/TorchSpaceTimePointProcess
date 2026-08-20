import copy
from pathlib import Path

import torch

from torchstpp import TorchSTPPConfig, load_config, run_etas

from examples.custom_etas3d import UserETAS3D, build_user_etas3d


def test_external_etas3d_runs_through_standard_pipeline():
    root = Path(__file__).resolve().parents[2]
    result = run_etas(
        root / "examples" / "custom_etas3d.toml",
        model_factories={"custom_etas3d": build_user_etas3d},
    )

    assert isinstance(result.run.model, UserETAS3D)
    assert len(result.run.model.parameter_values()) == 10
    assert result.run.catalog.depth is not None
    assert result.run.fit_result.outer_iterations == 1
    assert torch.isfinite(
        torch.as_tensor(result.run.fit_result.negative_log_likelihood)
    )


def test_external_etas3d_matches_the_builtin_reference():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "examples" / "custom_etas3d.toml")
    custom = run_etas(
        config,
        model_factories={"custom_etas3d": build_user_etas3d},
    )

    builtin_data = copy.deepcopy(dict(config.data))
    builtin_data["model"]["type"] = "etas3d"
    builtin = run_etas(TorchSTPPConfig(data=builtin_data, path=config.path))

    assert custom.run.fit_result.negative_log_likelihood == (
        builtin.run.fit_result.negative_log_likelihood
    )
    custom_parameters = custom.run.model.parameter_values()
    builtin_parameters = builtin.run.model.parameter_values()
    assert custom_parameters.keys() == builtin_parameters.keys()
    for name in custom_parameters:
        assert torch.equal(custom_parameters[name], builtin_parameters[name]), name

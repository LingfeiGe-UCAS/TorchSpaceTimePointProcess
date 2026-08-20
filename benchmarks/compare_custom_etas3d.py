"""Compare the external ETAS3D example against the built-in implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torchstpp import run_etas
from examples.custom_etas3d import build_user_etas3d


def _gradients(run):
    model = run.run.model
    model.zero_grad(set_to_none=True)
    loss = run.run.likelihood(
        model,
        run.run.catalog,
        run.run.domain,
        run.run.background_state,
    )
    loss.backward()
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return float(loss.detach().cpu()), gradients


def main(config: Path) -> None:
    builtin = run_etas(config, write_outputs=False)
    custom = run_etas(
        config,
        write_outputs=False,
        model_factories={"etas3d": build_user_etas3d},
    )

    builtin_parameters = builtin.run.model.parameter_values()
    custom_parameters = custom.run.model.parameter_values()
    parameter_differences = {
        name: abs(float(custom_parameters[name]) - float(reference))
        for name, reference in builtin_parameters.items()
    }
    parameter_relative_differences = {
        name: difference / max(abs(float(builtin_parameters[name])), 1e-300)
        for name, difference in parameter_differences.items()
    }

    builtin_loss, builtin_gradients = _gradients(builtin)
    custom_loss, custom_gradients = _gradients(custom)
    gradient_differences = {
        name: float(torch.max(torch.abs(custom_gradients[name] - reference)))
        for name, reference in builtin_gradients.items()
    }
    gradient_relative_differences = {
        name: difference
        / max(float(torch.max(torch.abs(builtin_gradients[name]))), 1e-300)
        for name, difference in gradient_differences.items()
    }

    result = {
        "builtin_nll": builtin.run.fit_result.negative_log_likelihood,
        "custom_nll": custom.run.fit_result.negative_log_likelihood,
        "absolute_nll_difference": abs(
            custom.run.fit_result.negative_log_likelihood
            - builtin.run.fit_result.negative_log_likelihood
        ),
        "recomputed_builtin_nll": builtin_loss,
        "recomputed_custom_nll": custom_loss,
        "maximum_absolute_parameter_difference": max(
            parameter_differences.values()
        ),
        "maximum_relative_parameter_difference": max(
            parameter_relative_differences.values()
        ),
        "maximum_absolute_gradient_difference": max(gradient_differences.values()),
        "maximum_relative_gradient_difference": max(
            gradient_relative_differences.values()
        ),
        "builtin_elapsed_seconds": builtin.run.elapsed_seconds,
        "custom_elapsed_seconds": custom.run.elapsed_seconds,
        "parameter_differences": parameter_differences,
        "gradient_differences": gradient_differences,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    main(arguments.config)

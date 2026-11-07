import pytest
import torch

from torchstpp.models.parameters import ModelParameter, ParameterConfig


def test_centered_log_preserves_initial_value_and_uses_it_as_scale():
    parameter = ModelParameter(
        ParameterConfig(
            191.63,
            lower_bound=0.0,
            transform="centered_log",
        )
    )

    assert parameter.transform == "centered_log"
    assert parameter.reference_scale == pytest.approx(191.63)
    assert parameter.raw.item() == pytest.approx(0.0)
    assert parameter().item() == pytest.approx(191.63)


def test_centered_log_has_dimensionless_exponential_gradient():
    parameter = ModelParameter(
        ParameterConfig(
            191.63,
            lower_bound=0.0,
            transform="centered_log",
        )
    )

    parameter().backward()

    assert parameter.raw.grad is not None
    assert parameter.raw.grad.item() == pytest.approx(191.63)


def test_centered_log_requires_one_sided_lower_bound():
    with pytest.raises(ValueError, match="requires a lower bound"):
        ModelParameter(ParameterConfig(2.0, transform="centered_log"))
    with pytest.raises(ValueError, match="requires a lower bound"):
        ModelParameter(
            ParameterConfig(
                2.0,
                lower_bound=0.0,
                upper_bound=3.0,
                transform="centered_log",
            )
        )

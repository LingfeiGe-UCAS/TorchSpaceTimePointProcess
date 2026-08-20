import torch

from torchstpp.models.backgrounds import GaussianKDEBackground
from torchstpp.models.etas import ETAS2D, ETAS2DParameters
from torchstpp.models.parameters import ParameterConfig


def test_user_defined_etas2d_parameters():
    parameters = ETAS2DParameters(
        mu=ParameterConfig(0.05, trainable=True),
        amplitude=ParameterConfig(0.20, trainable=True),
        alpha=ParameterConfig(1.10, trainable=True),
        c=ParameterConfig(0.02, trainable=False),
        p=ParameterConfig(1.20, trainable=True),
        d=ParameterConfig(0.60, trainable=True),
        q=ParameterConfig(1.60, trainable=True),
        gamma=ParameterConfig(0.90, trainable=True),
    )
    model = ETAS2D(parameters)

    assert torch.isclose(model.mu, torch.tensor(0.05))
    assert torch.isclose(model.p, torch.tensor(1.20))
    assert model.c_parameter.raw.requires_grad is False
    assert model.mu_parameter.raw.requires_grad is True
    assert model.mu > 0
    assert model.p > 1
    assert model.q > 1


def test_user_parameter_bounds_are_respected():
    defaults = ETAS2DParameters.defaults()
    parameters = ETAS2DParameters(
        mu=defaults.mu,
        amplitude=defaults.amplitude,
        alpha=defaults.alpha,
        c=defaults.c,
        p=ParameterConfig(1.2, lower_bound=1.1),
        d=defaults.d,
        q=ParameterConfig(1.959, lower_bound=1.0002, upper_bound=3.0),
        gamma=defaults.gamma,
    )
    model = ETAS2D(parameters)
    assert model.p_parameter.lower_bound == 1.1
    assert model.q_parameter.lower_bound == 1.0002
    assert model.q_parameter.upper_bound == 3.0
    assert torch.isclose(model.p, torch.tensor(1.2))
    assert torch.isclose(model.q, torch.tensor(1.959))


def test_default_parameter_bounds_are_available():
    model = ETAS2D()
    assert model.mu_parameter.lower_bound == 0.0
    assert model.p_parameter.lower_bound == 1.0
    assert model.q_parameter.lower_bound == 1.0


def test_etas2d_default_and_custom_background():
    model = ETAS2D()
    assert isinstance(model.background, GaussianKDEBackground)

    class CustomBackground:
        pass

    custom_background = CustomBackground()
    custom_model = ETAS2D(background=custom_background)
    assert custom_model.background is custom_background

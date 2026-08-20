from types import SimpleNamespace

import torch

from torchstpp.models import CustomPointProcessModel, ParameterConfig


def test_custom_etas3d_complete_interface():
    def background(target, params):
        return params["mu"].expand(target.time.shape[0])

    def trigger_kernel(target, source, params):
        dt = target.time - source.time
        distance = torch.linalg.vector_norm(
            target.position - source.position, dim=-1
        )
        return (
            params["amplitude"]
            * (dt + params["c"]).pow(-params["p"])
            * (distance.square() + params["d"]).pow(-params["q"])
        )

    def background_integral(domain, params):
        duration = domain.time_end - domain.time_start
        return params["mu"] * duration * domain.volume

    def source_integral(source, domain, params):
        # A deliberately simple differentiable example. Users may replace this
        # with an analytical formula, quadrature, or another numerical method.
        remaining_time = (domain.time_end - source.time).clamp_min(0.0)
        return params["amplitude"] * remaining_time

    model = CustomPointProcessModel(
        parameters={
            "mu": ParameterConfig(0.02, lower_bound=0.0),
            "amplitude": ParameterConfig(0.15, lower_bound=0.0),
            "c": ParameterConfig(0.01, lower_bound=0.0),
            "p": ParameterConfig(1.15, lower_bound=1.0),
            "d": ParameterConfig(0.5, lower_bound=0.0),
            "q": ParameterConfig(1.5, lower_bound=1.0),
        },
        background=background,
        trigger_kernel=trigger_kernel,
        background_integral=background_integral,
        source_integral=source_integral,
    )

    target = SimpleNamespace(
        time=torch.tensor([2.0]), position=torch.tensor([[1.0, 2.0, 3.0]])
    )
    source = SimpleNamespace(
        time=torch.tensor([1.0]), position=torch.tensor([[0.0, 0.0, 0.0]])
    )
    domain = SimpleNamespace(time_start=0.0, time_end=10.0, volume=8.0)

    assert torch.allclose(model.background_rate(target), torch.tensor([0.02]))
    assert model.pairwise_rate(target, source).shape == (1,)
    assert model.integrated_background(domain).ndim == 0
    assert model.integrated_source(source, domain).shape == (1,)
    assert model.parameter_values()["p"] > 1.0

    loss = (
        model.background_rate(target).sum()
        + model.pairwise_rate(target, source).sum()
        + model.integrated_background(domain)
        + model.integrated_source(source, domain).sum()
    )
    loss.backward()
    assert model.parameter_store["mu"].raw.grad is not None
    assert model.parameter_store["amplitude"].raw.grad is not None

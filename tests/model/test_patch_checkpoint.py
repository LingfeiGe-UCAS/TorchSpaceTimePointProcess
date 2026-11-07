import pytest
import torch
from tests.model.test_finite_source3d import make_model, source_catalog, target_catalog, DomainStub
from torchstpp.integration import PolygonSpatialIntegrator
from torchstpp.likelihood import ExactLikelihood
from unittest.mock import patch

@pytest.mark.parametrize('ids', [(0,7,9),(0,0,0)])
@pytest.mark.parametrize('polygon', [False,True])
def test_fine_checkpoint_value_gradient(ids, polygon):
    model = make_model(patch_chunk_size=1)
    if polygon:
        model.spatial_integrator = PolygonSpatialIntegrator(subdivisions=11, source_chunk_size=1)
    source, target = source_catalog(ids), target_catalog()
    domain = DomainStub(0.,4.,torch.tensor([[0.,0.],[2.,0.],[2.,2.],[0.,2.]],dtype=torch.float64))
    values = []
    for enabled in (False,True):
        model.checkpoint_patches = enabled
        model.zero_grad(set_to_none=True)
        loss = model.pairwise_rate(target,source).sum()+model.integrated_source(source,domain).sum()+model.mu
        loss.backward()
        values.append((loss.detach(), {n:p.grad.clone() if p.grad is not None else None for n,p in model.named_parameters()}))
    torch.testing.assert_close(values[0][0], values[1][0], rtol=1e-12,atol=1e-12)
    for key in values[0][1]:
        torch.testing.assert_close(values[0][1][key],values[1][1][key],rtol=1e-11,atol=1e-12)

def test_no_outer_checkpoint():
    model = make_model()
    model.checkpoint_patches = True
    with patch('torchstpp.likelihood.exact.checkpoint', side_effect=AssertionError('nested checkpoint')):
        ExactLikelihood(checkpoint_blocks=True)._run_block(model, lambda: model.mu)

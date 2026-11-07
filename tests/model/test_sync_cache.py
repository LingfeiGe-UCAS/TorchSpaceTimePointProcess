import pytest
import torch
from torchstpp.models.components.fixed_inputs import FixedInputCache
from torchstpp.models.components.depth import conditional_beta_depth_density

def test_partition_rechecks_mutated_ids():
    from tests.model.test_finite_source3d import make_model, source_catalog
    model = make_model()
    catalog = source_catalog()
    first = model.finite_source_parent_indices(catalog)
    assert model.finite_source_parent_indices(catalog) is first
    catalog.finite_source_id[0] = -1
    with pytest.raises(ValueError, match='non-negative'):
        model.finite_source_parent_indices(catalog)

def test_views_and_mutation():
    cache = FixedInputCache(2)
    t = torch.arange(5.)
    calls = []
    def build():
        calls.append(1)
        return len(calls)
    assert cache.get_or_create((t[:2],), (), build) == 1
    assert cache.get_or_create((t[:2],), (), build) == 1
    t[0] = 8
    assert cache.get_or_create((t[:2],), (), build) == 2
    assert cache.get_or_create((t[1:3],), (), build) == 3
    assert len(cache.entries) == 2

def test_depth_rechecks_modified_data_and_parameters():
    target = torch.tensor([1.])
    source = torch.tensor([2.])
    concentration = torch.tensor(3., requires_grad=True)
    def run():
        return conditional_beta_depth_density(target, source, 30., concentration)
    run().sum().backward()
    assert concentration.grad is not None
    target[0] = -1
    with pytest.raises(ValueError, match='target depth'):
        run()
    target[0] = 1
    with torch.no_grad():
        concentration.fill_(-1)
    with pytest.raises(ValueError, match='concentration'):
        run()

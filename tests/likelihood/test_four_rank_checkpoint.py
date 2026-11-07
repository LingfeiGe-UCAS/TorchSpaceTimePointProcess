from datetime import timedelta
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tests.model.test_finite_source3d import make_model, source_catalog
from tests.likelihood.test_exact import DomainStub
from torchstpp.models.backgrounds import KDEState
from torchstpp.integration import PolygonSpatialIntegrator
from torchstpp.likelihood import ExactLikelihood, DistributedExactLikelihood
from torchstpp.inference import FortranEMLBFGS, LegacyEMLBFGSConfig

def problem():
    m = make_model(integrator=PolygonSpatialIntegrator(subdivisions=11, source_chunk_size=1))
    m.checkpoint_patches = True
    c = source_catalog()
    d = DomainStub(0.,0.,4.,torch.tensor([[0.,0.],[2.,0.],[2.,2.],[0.,2.]],dtype=torch.float64))
    s = KDEState(position=c.position.clone(), depth=c.depth.clone(),
                 bandwidth=torch.full((3,),.5,dtype=torch.float64),
                 weights=torch.full((3,),.5,dtype=torch.float64),duration=4.)
    return m,c,d,s

def fitter():
    return FortranEMLBFGS(LegacyEMLBFGSConfig(outer_iterations=2,e_steps_per_outer=1,
        decrease_e_steps=False,lbfgs_max_iter=2,line_search_fn=None,
        target_chunk_size=1,source_chunk_size=1))

def run(distributed, mode):
    m,c,d,s = problem()
    exact = ExactLikelihood(target_chunk_size=1,source_chunk_size=1,checkpoint_blocks=True)
    l = DistributedExactLikelihood(exact,pair_budget=2 if mode=='pair' else None,source_alignment=1) if distributed else exact
    if distributed:
        terms = l.terms(m,c,d,s)
        l.backward_and_sum_gradients_(terms.backward_loss,m)
        loss = terms.global_terms.negative_log_likelihood
    else:
        loss = l(m,c,d,s)
        loss.backward()
    gradients = {n:p.grad.clone() for n,p in m.named_parameters()}
    result = fitter().fit(m,l,c,d,s)
    return dict(loss=loss.detach(),gradients=gradients,phi=result.background_probability,
                tau=m.patch_tau,parameters={n:p.detach().clone() for n,p in m.named_parameters()})

def worker(rank, rendezvous, out, mode):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=rendezvous,rank=rank,world_size=4,timeout=timedelta(seconds=90))
    try:
        torch.save(run(True,mode),Path(out)/f'{rank}.pt')
    finally:
        dist.destroy_process_group()

@pytest.mark.parametrize('mode',['cyclic','pair'])
def test_four_ranks_with_empty_work(tmp_path,mode):
    reference = run(False,mode)
    mp.spawn(worker,args=((tmp_path/'rdzv').resolve().as_uri(),str(tmp_path),mode),nprocs=4,join=True)
    for rank in range(4):
        actual=torch.load(tmp_path/f'{rank}.pt',weights_only=True)
        torch.testing.assert_close(actual,reference,rtol=1e-9,atol=1e-10)

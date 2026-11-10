"""Synthetic-only demonstration. No dataset, checkpoint, seed-search or retry options."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from experiments.node_dp_slots.config import Spec
from experiments.node_dp_slots.adapter import Graph
from experiments.node_pdp.registry import Registry
from .mechanism import Engine
from . import diagnostics,policy,accounting


def counterexample():
    """Zero hidden messages despite128 active nodes and strictly positive w3."""
    import igraph
    from experiments.node_dp_slots.mechanism import new_model
    from experiments.graph_util import get_rev_dgl
    from baselines.heuristics import get_influence
    model=new_model()
    with torch.no_grad():
        for layer in (model.grat.grat1,model.grat.grat2,model.grat.grat3):
            layer.fc.weight.zero_();layer.fc.bias.fill_(-1.)
            layer.attn_fc.weight.zero_();layer.attn_fc.bias.zero_()
        model.grat.grat3.fc.weight.fill_(1.);model.grat.grat3.fc.bias.fill_(.25)
    policy.project(model,policy.Layout.from_model(model));model.eval()
    edges=[(50,v) for v in range(128) if v!=50]+[(v,50) for v in range(51,128)]
    graph=igraph.Graph(n=128,edges=edges,directed=True)
    message=get_rev_dgl(graph,'ones',32,True,False)
    with torch.no_grad():logits=model.logits(message,message.ndata['feat']).reshape(-1)
    chosen=torch.argsort(logits,descending=True,stable=True)[:50].tolist()
    witness=[50]+list(range(49))  # Fixed by construction, no optimization/search.
    return dict(scope='artificial counterexample only',nodes=128,k=50,positive_last_weights=32,
                all_logits_bias=bool((logits==.25).all()),selected_zero_outdegree=graph.outdegree(chosen).count(0),
                projected_spread=get_influence(graph,chosen),fixed_witness_spread=get_influence(graph,witness),
                proves='nonnegative margin is not strict margin or IM utility guarantee')


def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError('new synthetic run-id required')
    output.mkdir(parents=True,mode=0o700);output.chmod(0o700)
    restricted=output/'restricted';restricted.mkdir(mode=0o700)
    previous=os.umask(0o077)
    def save(name,data):
        with (restricted/name).open('x') as handle:json.dump(data,handle,indent=2,sort_keys=True,allow_nan=False)
    try:
        torch.set_num_threads(1)
        U=tuple(range(160))
        reg=Registry(U,tuple((v,(1,2,4,2)[v%4]) for v in U),source='public-artificial-v2-synthetic')
        spec=Spec(U)
        graph=Graph(tuple(range(128)),tuple((v,(v+k)%128) for v in range(128) for k in range(1,17)),True)
        result=Engine.testing(reg,spec,sampling_seed=0,noise_seed=0).run(graph)
        d=result.testing_diagnostics
        assert d['updates']==d['noise_calls']==d['projection_calls']==100
        assert d['all_projected_weights_nonnegative'] and d['all_parameters_finite']
        save('testing_diagnostics.json',d)
        save('public_synthetic_configuration.json',dict(protocol='node_pdp_v2_projected',base=spec.public(),
            registry=reg.public(),projection=result._layout.public(),testing_seeds=dict(sampling=0,gaussian=0),
            testing_only=True,export_permitted=False,machine_dp_certified=False,machine_pdp_certified=False))
        example=counterexample();assert example['projected_spread']==50 and example['fixed_witness_spread']==128
        save('counterexample.json',example)
        save('diagnostic_definitions.json',diagnostics.definitions())
        save('accounting.json',dict(base=accounting.base_account(spec),nodes=accounting.node_accounts(reg,spec)))
        save('source_sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')})
        save('completion.json',dict(status='completed',scope='artificial graphs/parameters only',updates=100,
            real_prepared_access=False,real_model_access=False,synthetic_counterexample_spread=50,
            machine_dp_certified=False,machine_pdp_certified=False,development_process_dp_protected=False))
    finally:os.umask(previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)


if __name__=='__main__':main()

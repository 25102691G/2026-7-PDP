"""Restricted scientific diagnostics using original ranking and IC functions."""
import torch


def prepare_evaluation_graph(graph):
    """Original reversed GRAT messages and outgoing IC graph, without N-by-N data."""
    from experiments import node_sampling_utility as u
    rt = u.runtime()
    message = rt['graph_util'].get_rev_dgl(graph, 'ones', 32, graph.is_directed(), False)
    coverage, message = rt['load_graph'].connect_node_to_graph(list(range(graph.vcount())), message)
    u.check_message_graph(graph, message, coverage)
    return message, coverage


def evaluate(model,graph,message,reference,k=50):
    from experiments import node_sampling_utility as u
    from baselines.heuristics import get_influence
    rt=u.runtime();probe={}
    def capture(_module,args):probe['m']=args[0].detach().clone()
    hook=model.grat.grat3.fc.register_forward_pre_hook(capture);model.eval()
    try:
        with torch.no_grad():scores=rt['evaluate'].ranking_scores(model,message,message.ndata['feat']).reshape(-1)
    finally:hook.remove()
    if not bool(torch.isfinite(scores).all()):raise RuntimeError('nonfinite test logits')
    selected=rt['evaluate'].select_topk_nodes(scores,k).tolist()
    if len(set(selected))!=k:raise RuntimeError('invalid seed set')
    spread=get_influence(graph,selected);independent=set(selected)
    for x,y in graph.get_edgelist():
        if x in selected:independent.add(y)
    if spread!=len(independent):raise RuntimeError('outgoing one-step IC mismatch')
    degrees=torch.tensor(graph.outdegree());active=degrees>0
    bias=model.grat.grat3.fc.bias.detach().double().reshape(())
    margin=scores.double()-bias;m=probe['m'];weights=model.grat.grat3.fc.weight.detach()
    _,counts=torch.unique(scores,return_counts=True);boundary=torch.sort(scores,descending=True).values[k-1]
    nplus=int(((margin>0)&active).sum())
    return dict(spread=spread,coverage=None if reference is None else spread/reference*100,reference_spread=reference,n=1,std=None,
        topk_local_ids=selected,selected_zero_outdegree=int((degrees[selected]==0).sum()),
        N_plus=nplus,N_plus_ge_k=nplus>=k,nonzero_outdegree_nodes=int(active.sum()),
        nonzero_degree_zero_margin=int(((margin==0)&active).sum()),nonzero_degree_negative_margin=int(((margin<0)&active).sum()),
        last_zero_aggregate_nodes=int((m==0).all(1).sum()),last_zero_aggregate_fraction=float((m==0).all(1).double().mean()),
        aggregate_nonnegative=bool((m>=0).all()),last_zero_weights=int((weights==0).sum()),
        last_weights_nonnegative=bool((weights>=0).all()),bias_tie_nodes=int((margin==0).sum()),
        unique_logits=len(counts),max_tie_count=int(counts.max()),boundary_tie_count=int((scores==boundary).sum()),
        boundary_tie_crosses_k=bool((scores>=boundary).sum()>k),all_logits_finite=True)

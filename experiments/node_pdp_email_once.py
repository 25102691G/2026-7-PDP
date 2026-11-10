"""One specifically authorized Email attempt, with observational checks only.

No alternative run-id, seed, configuration or retry CLI. All artifacts restricted.
Imports never load a prepared graph or launch training.
"""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/node_pdp/email-first-real-20261031-v1/restricted'
LABELS = ['idealized-only', 'exploratory', 'machine-uncertified']


def save(path, obj):
    with Path(path).open('x') as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')
    Path(path).chmod(0o600)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reserve_attempt(directory):
    save(Path(directory)/'training_attempt.json', dict(
        maximum_real_training_attempts=1, state='authorization_consumed_before_execution',
        pid=os.getpid(), timestamp=datetime.now(timezone.utc).isoformat(), labels=LABELS))


def load_frozen(directory):
    from experiments.node_pdp.registry import Registry
    from experiments.node_dp_slots.config import Spec
    from experiments.node_pdp.accounting import node_accounts
    directory=Path(directory)
    stored=json.loads((directory/'registry.json').read_text())
    data=stored['registry']
    registry=Registry(tuple(data['universe']),tuple((r['identity'],r['epsilon_target']) for r in data['entries']),
                      source=data['source'],version=data['version'])
    freeze=json.loads((directory/'freeze.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text())
    digest=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if not (registry.public()==data and registry.fingerprint==stored['fingerprint']==freeze['registry_fingerprint']
            ==protocol['registry_fingerprint'] and digest==freeze['protocol_sha256'] and freeze['public_U_source_verified']):
        raise ValueError('public registry/protocol seal mismatch')
    spec=Spec(registry.universe,True)
    if protocol['base']!=spec.public():raise ValueError('frozen base changed')
    node_accounts(registry,spec)
    return registry,spec,protocol


def audited_run(registry, spec, graph):
    """Run genuine privacy Engine once. Never retain a noise draw or raw gradient."""
    import torch
    from experiments.node_pdp import mechanism as outer, sampling
    from experiments.node_dp_slots import mechanism as base, noise
    engine=outer.Engine.privacy(registry,spec)
    if (engine._mode!='privacy' or type(engine._sampler) is not sampling.SecretBernoulli
        or type(engine._base._source) is not noise.SecretGaussian):
        raise PermissionError('only actual production random sources allowed')
    audit=dict(labels=LABELS,outer_sample_calls=0,induced_graph_calls=0,adapter_calls=0,
               updates=0,noise_calls=0,noise_coordinates=0,double_clip_calls=0,
               per_position_clip_calls=0,slot_clip_trigger_count=0,all_parameters_finite=True,
               source_types=['SecretBernoulli','SecretGaussian'],steps=[],s_sum=260.,denominator=16)
    original_sample=sampling.sample_once;original_induce=sampling.induce
    original_adapter=base.Adapter;original_batch=base.Adapter.batch
    original_clip=base.clip;original_double=base.double_clip
    original_update=base.update;original_noise=noise.SecretGaussian.draw
    original_model=base.new_model
    pending=[None]

    def new_model():
        model=original_model()
        def finite_forward(_module,_args,result):
            if not bool(torch.isfinite(result).all()):
                raise RuntimeError('nonfinite training forward; stop without retry')
        model.register_forward_hook(finite_forward)
        return model

    def sample(*args):
        audit['outer_sample_calls']+=1
        if audit['outer_sample_calls']!=1:raise RuntimeError('outer sampling repeated')
        return original_sample(*args)

    def induce(*args):
        audit['induced_graph_calls']+=1
        return original_induce(*args)

    def adapter(*args):
        audit['adapter_calls']+=1
        result=original_adapter(*args)
        audit['container_counts']=[len(c.subgraphs) for c in result.caches]
        return result

    def batch(instance):
        result=original_batch(instance)
        if [len(s) for s in result]!=[4]*4:raise RuntimeError('fixed quota changed')
        counts=[sum(item is not None for item in slot) for slot in result]
        audit['steps'].append(dict(positions=16,effective_positions=sum(counts),
                                  zero_slots=sum(n==0 for n in counts)))
        return result

    def clip(vector):
        result=original_clip(vector)
        if float(result.norm())>3.:raise RuntimeError('per-position clipping bound failed')
        audit['per_position_clip_calls']+=1
        return result

    def double(vectors):
        result,slots=original_double(vectors)
        if tuple(vectors.shape)!=(4,4,2340) or any(float(s.norm())>3. for s in slots):
            raise RuntimeError('slot clipping bound failed')
        audit['slot_clip_trigger_count']+=sum(float(v.sum(0).norm())>3. for v in vectors)
        audit['double_clip_calls']+=1
        return result,slots

    def draw(source):
        z=original_noise(source)
        if z.shape!=(2340,) or not bool(torch.isfinite(z).all()):raise RuntimeError('nonfinite noise')
        # Check the exact pre-totalization update for overflow, without saving z.
        params,signal=pending[0]
        previous=torch.cat([p.detach().reshape(-1).double() for p in params])
        if not bool(torch.isfinite((previous-.005*(signal.double()+z.double())/16).float()).all()):
            raise RuntimeError('nonfinite pre-repair model state')
        audit['noise_calls']+=1;audit['noise_coordinates']+=z.numel()
        return z

    def update(parameters, signal, source):
        params=list(parameters)
        if sum(p.numel() for p in params)!=2340 or not all(bool(torch.isfinite(p).all()) for p in params):
            raise RuntimeError('invalid pre-update model')
        before=[p.detach().clone() for p in params]
        pending[0]=(params,signal)
        calls=audit['noise_calls']
        try:original_update(params,signal,source)
        finally:pending[0]=None
        if audit['noise_calls']!=calls+1:raise RuntimeError('noise call count changed')
        finite=all(bool(torch.isfinite(p).all()) for p in params)
        audit['all_parameters_finite'] &= finite
        if not finite:raise RuntimeError('nonfinite model')
        audit['steps'][-1]['parameters_changed']=any(not torch.equal(p,b) for p,b in zip(params,before))
        audit['updates']+=1

    with ExitStack() as stack:
        for owner,name,replacement in ((sampling,'sample_once',sample),(sampling,'induce',induce),
            (original_adapter,'batch',batch),(base,'Adapter',adapter),(base,'clip',clip),
            (base,'double_clip',double),(base,'update',update),(noise.SecretGaussian,'draw',draw),
            (base,'new_model',new_model)):
            stack.enter_context(patch.object(owner,name,replacement))
        result=engine.run(graph)
    if not (audit['updates']==audit['noise_calls']==audit['double_clip_calls']==len(audit['steps'])==100
            and audit['outer_sample_calls']==audit['induced_graph_calls']==audit['adapter_calls']==1):
        raise RuntimeError('fixed execution audit failed')
    audit['base_ledger_models']=engine._base._ledger._ledger.models
    if audit['base_ledger_models']!=1:raise RuntimeError('base budget reservation mismatch')
    return result,audit


def restore_model(values):
    import torch
    from experiments.node_dp_slots.mechanism import new_model
    if len(values)!=2340 or not all(math.isfinite(v) for v in values):raise ValueError('invalid parameters')
    model=new_model()
    flat=torch.tensor(values,dtype=torch.float32)
    with torch.no_grad():
        for p,part in zip(model.parameters(),flat.split([p.numel() for p in model.parameters()])):
            p.copy_(part.reshape(p.shape))
    return model


def evaluate(model, graph, label):
    from experiments import node_sampling_utility as u
    rt=u.runtime();torch=rt['torch'];dg,coverage,_=u.prepare_message_graph(graph)
    result=u.evaluate(model,coverage,dg,432,0,label)
    with torch.no_grad():
        scores=rt['evaluate'].ranking_scores(model,dg,dg.ndata['feat']).reshape(-1)
        selected=rt['evaluate'].select_topk_nodes(scores,50).tolist()
        _,counts=torch.unique(scores,return_counts=True)
        boundary=torch.sort(scores,descending=True).values[49]
    result.update(top50_local_ids=selected,selected_zero_outdegree_count=coverage.outdegree(selected).count(0),
                  max_logit_tie_count=int(counts.max()),unique_logits=len(counts),
                  boundary_tie_count=int((scores==boundary).sum()),
                  boundary_tie_crosses_k=bool((scores>=boundary).sum()>50),
                  test_privacy_status='restricted; not claimed DP/public output')
    return result


def execute():
    """Only the fixed approved run; any attempt file prevents another process retry."""
    os.umask(0o077)
    registry,spec,protocol=load_frozen(RUN)
    gate=json.loads((RUN/'input_gate.json').read_text())
    preflight=json.loads((RUN/'preflight.json').read_text())
    if gate['status']!='passed' or not preflight['passed']:raise RuntimeError('gates not passed')
    for path,digest in gate['input_hashes'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('prepared input changed')
    reserve_attempt(RUN)
    from experiments import node_sampling_utility as u
    from experiments.node_sampling_structure import load_prepared_train
    from experiments.node_dp_slots.adapter import Graph
    from experiments.node_dp_slots import mechanism as base
    from experiments.node_pdp import release,accounting
    started=time.perf_counter()
    try:
        rt=u.runtime();rt['torch'].set_num_threads(1)
        g,_,_=load_prepared_train(ROOT/'dataset/prepared','email')
        ids=tuple(g.vs['original_id'])
        graph=Graph(ids,tuple((ids[u],ids[v]) for u,v in g.get_edgelist()),True)
        _,test,directed=rt['loader'].load_prepared_graph('email','test',ROOT/'dataset/prepared')
        if not directed or rt['evaluate'].calculate_celf_coverages(test,1,[50])[50]!=432:
            raise RuntimeError('test protocol mismatch')
        initial=evaluate(base.new_model(),test,'initial')
        save(RUN/'initial_evaluation.json',initial)
        training_start=time.perf_counter()
        result,audit=audited_run(registry,spec,graph)
        audit['training_seconds']=time.perf_counter()-training_start
        save(RUN/'execution_audit.json',audit)
        try:release.export(result)
        except PermissionError:strict_blocked=True
        else:raise RuntimeError('strict export unexpectedly enabled')
        artifact=release.export(result,permit_idealized=True)
        save(RUN/'idealized_model.json',artifact)
        final=evaluate(restore_model(artifact['parameters']),test,'node-pdp-final')
        final['n']=1;final['sample_standard_deviation']=None
        save(RUN/'final_evaluation.json',final)
        save(RUN/'completion.json',dict(status='completed',real_training_attempts=1,labels=LABELS,
            updates=audit['updates'],strict_export_blocked=strict_blocked,machine_dp_certified=False,
            machine_pdp_certified=False,registry_fingerprint=registry.fingerprint,
            base_account=accounting.base_account(spec),node_accounts=accounting.node_accounts(registry,spec),
            elapsed_seconds=time.perf_counter()-started,initial_final_parameters_differ=any(s['parameters_changed'] for s in audit['steps'])))
    except BaseException as error:
        save(RUN/'execution_failure.json',dict(status='stopped-no-retry',labels=LABELS,
             exception_type=type(error).__name__,real_training_attempts_at_most=1,
             message='UNRESOLVED execution gap; no retry authorized'))
        raise


if __name__=='__main__':
    with (RUN/'execution.log').open('x') as log,redirect_stdout(log),redirect_stderr(log):
        execute()

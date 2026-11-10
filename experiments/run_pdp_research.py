"""Paper-style Node-PDP v1/v2 research CLI; no production publication control."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
PREPARED=ROOT/'dataset/prepared'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,sort_keys=True,allow_nan=False)
    Path(path).chmod(0o600)


def datasets():
    return sorted(p.name for p in PREPARED.iterdir() if p.is_dir() and (p/'metadata.json').is_file())


def prepared_universe(mapping_path):
    """Declared fixed domain from ALL prepared mapping rows, never V_train alone."""
    with Path(mapping_path).open() as handle:
        rows = list(csv.DictReader(handle))
    identities = [int(row['original_id']) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate stable identity in prepared mapping')
    for split in ('train', 'test'):
        local = [int(row['local_id']) for row in rows if row['split'] == split]
        if sorted(local) != list(range(len(local))):
            raise ValueError('prepared local mapping is not a bijection')
    if any(row['split'] not in ('train', 'test') for row in rows):
        raise ValueError('unknown prepared split')
    return tuple(sorted(identities))


def registry_from_prepared_mapping(mapping_path, *, public_seed):
    """User-declared public domain; does NOT certify preparation or its disclosure."""
    from experiments.node_pdp.registry import Registry
    return Registry.from_public_quotas(prepared_universe(mapping_path), public_seed=public_seed,
        source='user-declared fixed prepared mapping domain; original preparation not DP-certified')


def load_inputs(dataset,registry_path,reference):
    if registry_path is None:raise ValueError('external public Registry required; never infer U/budgets from private V')
    if dataset not in datasets() or (reference is not None and (not math.isfinite(reference) or reference<=0)):
        raise ValueError('existing dataset and positive reference or None required')
    from experiments.node_pdp.registry import Registry
    from experiments.node_dp_slots.config import Spec
    from experiments.node_dp_slots.adapter import Graph
    from experiments.node_pdp.accounting import validate_public
    from experiments.node_sampling_structure import load_prepared_train
    from experiments import node_sampling_utility as u
    stored=json.loads(Path(registry_path).read_text());public=stored['registry']
    reg=Registry(tuple(public['universe']),tuple((e['identity'],e['epsilon_target']) for e in public['entries']),source=public['source'],version=public['version'])
    if reg.public()!=public or reg.fingerprint!=stored['fingerprint']:raise ValueError('frozen public registry mismatch')
    directory=PREPARED/dataset;meta=json.loads((directory/'metadata.json').read_text())
    if meta['train_ratio']!=.5:raise ValueError('original50:50 split required')
    domain_scope=stored.get('domain_scope','external-fixed-domain')
    if domain_scope=='fixed-prepared-mapping' and reg.universe!=prepared_universe(directory/'node_mapping.csv'):
        raise ValueError('frozen registry does not cover the complete declared prepared domain')
    spec=Spec(reg.universe,bool(meta['directed']));validate_public(reg,spec)
    rt=u.runtime();g,_,_=load_prepared_train(PREPARED,dataset)
    ids=tuple(g.vs['original_id']);train=Graph(ids,tuple((ids[x],ids[y]) for x,y in g.get_edgelist()),spec.directed)
    _,test,directed=rt['loader'].load_prepared_graph(dataset,'test',PREPARED)
    with (directory/'test_nodes.csv').open() as f:rows=list(csv.DictReader(f))
    mapped={int(r['local_id']):int(r['original_id']) for r in rows}
    if set(mapped)!=set(range(test.vcount())) or len(set(mapped.values()))!=len(mapped):raise ValueError('test ID mapping invalid')
    test_ids=tuple(mapped[i] for i in range(test.vcount()))
    if directed!=spec.directed or set(ids)&set(test_ids) or not(set(ids)|set(test_ids))<=set(reg.universe):raise ValueError('direction/domain/split mismatch')
    evidence=dict(dataset=dataset,prepared_split_unchanged=True,stable_ids_checked=True,train_test_disjoint=True,
        domain_scope=domain_scope,registry_fingerprint=reg.fingerprint,preparation_privacy_covered=False)
    start=time.perf_counter()
    if reference is None:
        from timeout_decorator import timeout, TimeoutError
        try:
            candidate=timeout(seconds=300)(rt['evaluate'].calculate_celf_coverages)(test,1,[50])[50]
            reference=float(candidate) if math.isfinite(candidate) and candidate>0 else None
        except TimeoutError:
            reference=None
        evidence['celf_reference_source']='existing evaluate_models.calculate_celf_coverages on prepared test, d=1,k=50; 300-second cap'
    else:
        evidence['celf_reference_source']='caller-supplied reference'
    evidence.update(celf_reference=reference,celf_seconds=time.perf_counter()-start)
    from experiments.pdp_research_evaluation import prepare_evaluation_graph
    dg,coverage=prepare_evaluation_graph(test)
    if dg.ndata[rt['dgl'].NID].tolist()!=list(range(test.vcount())):raise ValueError('test node ordering changed')
    inputs={str(p.relative_to(ROOT)):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for p in directory.iterdir() if p.is_file()}
    return reg,spec,train,coverage,dg,test_ids,inputs,evidence


def run(args):
    from experiments.node_pdp_research import Session
    from experiments.pdp_research_evaluation import evaluate
    from experiments import node_sampling_utility as u
    from experiments.node_dp_slots import mechanism as base
    from experiments.node_pdp import mechanism as prod
    from experiments.node_pdp.accounting import base_account
    from experiments.pdp_research_observer import TrainingObserver
    import torch
    torch.set_num_threads(1)
    reg,spec,graph,test,message,test_ids,inputs,evidence=load_inputs(args.dataset,args.registry,args.reference)
    output=Path(args.output).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('results must stay under outputs, never dataset')
    output.mkdir(parents=True,exist_ok=False,mode=0o700);output.chmod(0o700)
    restricted=output/'restricted';restricted.mkdir(mode=0o700)
    reference=evidence['celf_reference']
    config=dict(algorithm=args.algorithm,dataset=args.dataset,model_seed=args.model_seed,base={**spec.public(),'model_seed':args.model_seed},
        registry_fingerprint=reg.fingerprint,input_file_metadata=inputs,domain_scope=evidence['domain_scope'],
        preparation_privacy_covered=False,
        labels=['exploratory','idealized-only','machine-uncertified'],purpose='research',
        evaluation=dict(ic_p=1,steps=1,k=50,celf_reference=reference),publication_allowed=False)
    save(restricted/'configuration.json',config);save(restricted/'input_checks.json',evidence)
    save(restricted/'run_started.json',dict(status='started',n=1,std=None,no_automatic_retry=True))
    try:
        session=Session(reg,spec,args.algorithm,model_seed=args.model_seed)
        initial_hash=u.state_hash(session.model);start=time.perf_counter()
        initial=evaluate(session.model,test,message,reference);initial_seconds=time.perf_counter()-start
        save(restricted/'initial_evaluation.json',initial)
        guard,ledger=prod._PROCESS_GUARD,base._PROCESS_LEDGER;before=(guard._used,ledger.models)
        start=time.perf_counter()
        with TrainingObserver() as observer:
            result=session.run(graph)
        elapsed=time.perf_counter()-start
        save(restricted/'sampling_diagnostics.json',observer.summary())
        start=time.perf_counter();final=evaluate(session.model,test,message,reference);final_seconds=time.perf_counter()-start
        final['topk_original_ids']=[test_ids[i] for i in final['topk_local_ids']]
        if guard is not prod._PROCESS_GUARD or ledger is not base._PROCESS_LEDGER or before!=(guard._used,ledger.models):raise RuntimeError('production controls changed')
        for name,h in inputs.items():
            p=ROOT/name
            if dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)!=h:raise RuntimeError('prepared input changed')
        save(restricted/'final_evaluation.json',final);save(restricted/'training_diagnostics.json',result.diagnostics)
        save(restricted/'model.json',dict(parameters=list(result.parameters),configuration=result.configuration,publication_allowed=False))
        save(restricted/'completion.json',dict(status='completed',real_training_attempts=1,n=1,std=None,updates=result.diagnostics['updates'],
            training_seconds=elapsed,initial_evaluation_seconds=initial_seconds,final_evaluation_seconds=final_seconds,
            celf_seconds=evidence['celf_seconds'],evaluation_seconds=initial_seconds+final_seconds+evidence['celf_seconds'],
            initial_model_sha256=initial_hash,final_model_sha256=u.state_hash(session.model),
            parameters_changed=initial_hash!=u.state_hash(session.model),production_controls_unchanged=True,
            base_account_reference=base_account(spec),machine_pdp_certified=False,
            development_process_dp_protected=False))
    except BaseException as error:
        save(restricted/'failure.json',dict(status='failed-no-retry',exception_type=type(error).__name__,n=1,std=None))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=datasets(),required=True)
    parser.add_argument('--algorithm',choices=['node_pdp_v1','node_pdp_v2_projected'],required=True)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--model-seed',type=int,default=0)
    parser.add_argument('--reference',type=float,default=None,help='optional known CELF; otherwise compute existing deterministic reference')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();os.umask(0o077);run(args)


if __name__=='__main__':main()

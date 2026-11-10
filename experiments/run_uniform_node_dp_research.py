"""G0 Uniform research CLI. Real use requires a separately approved experiment stage."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from experiments.run_pdp_research import ROOT,load_inputs,save
from experiments.uniform_node_dp.config import Spec
from experiments.uniform_node_dp.research import Session
from experiments.pdp_research_evaluation import evaluate
from experiments.node_sampling_utility import state_hash

DATASETS=('email','bitcoin','lastfm','hepph','facebook')
REFERENCES=dict(email=432,bitcoin=1684,lastfm=1269,hepph=1741,facebook=3157)


def run(args):
    # The registry is reused for the frozen identity domain, never its heterogeneous probabilities.
    if args.dataset not in DATASETS:raise ValueError('five frozen datasets only')
    output=Path(args.output).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('new output must be under outputs')
    if output.exists():raise FileExistsError('no overwrite or retry')
    reg,domain,train,test,message,test_ids,inputs,evidence=load_inputs(args.dataset,args.registry,REFERENCES[args.dataset])
    spec=Spec(reg.universe,domain.directed)
    output.mkdir(parents=True,exist_ok=False,mode=0o700);output.chmod(0o700)
    restricted=output/'restricted';restricted.mkdir(mode=0o700)
    digests={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs}
    config=dict(algorithm='uniform_node_dp',dataset=args.dataset,model_seed=args.model_seed,
        base={**spec.public(),'model_seed':args.model_seed},registry_fingerprint=reg.fingerprint,
        input_sha256=digests,domain_scope=evidence['domain_scope'],preparation_privacy_covered=False,
        evaluation=dict(ic_p=1,steps=1,k=50,celf_reference=REFERENCES[args.dataset]),
        publication_allowed=False,purpose='research',labels=['idealized-only','machine-uncertified'])
    save(restricted/'configuration.json',config);save(restricted/'input_checks.json',evidence)
    save(restricted/'run_started.json',dict(status='started',n=1,std=None,no_automatic_retry=True))
    try:
        import torch
        torch.set_num_threads(1)
        session=Session(spec,model_seed=args.model_seed);initial=state_hash(session.model)
        start=time.perf_counter();result=session.run(train);elapsed=time.perf_counter()-start
        start=time.perf_counter();final=evaluate(session.model,test,message,REFERENCES[args.dataset]);eval_seconds=time.perf_counter()-start
        final['topk_original_ids']=[test_ids[i] for i in final['topk_local_ids']]
        if any(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h for n,h in digests.items()):
            raise RuntimeError('prepared file changed')
        save(restricted/'training_diagnostics.json',result.diagnostics)
        save(restricted/'model.json',dict(parameters=list(result.parameters),configuration=result.configuration,publication_allowed=False))
        save(restricted/'final_evaluation.json',final)
        save(restricted/'completion.json',dict(status='completed',real_training_attempts=1,n=1,std=None,
            updates=100,training_seconds=elapsed,final_evaluation_seconds=eval_seconds,
            initial_model_sha256=initial,final_model_sha256=state_hash(session.model),
            account=result.configuration['account'],machine_dp_certified=False,development_process_dp_protected=False))
    except BaseException as error:
        save(restricted/'failure.json',dict(status='failed-no-retry',exception_type=type(error).__name__,n=1,std=None))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--model-seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.model_seed<0:parser.error('nonnegative public model seed required')
    os.umask(0o077);run(args)


if __name__=='__main__':main()

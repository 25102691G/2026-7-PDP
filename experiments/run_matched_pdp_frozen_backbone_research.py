"""Independent prepared-data scientific runner; frozen33D Session is unchanged.

One output directory is an irrevocable attempt reservation. No retries,
production release, ledger charges, initial evaluation or new CELF computation.
"""
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.run_pdp_research import load_inputs

REFERENCES = dict(email=432,bitcoin=1684,lastfm=1269,hepph=1741,facebook=3157)
ALGORITHM = 'matched_pdp_frozen_backbone_v1'
BUDGET_ROOTS = ()
SAVE_LOCK = threading.RLock()


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def artifact_bytes(roots):
    return sum(f.stat().st_size for root in roots for f in Path(root).rglob('*') if f.is_file())


def resource_violation(task,wall,rss,disk):
    return wall>task['max_wall_minutes']*60 or rss>task['max_rss_gib']*2**30 or disk>2*2**30


def save(path, value, *, budget_roots=None, max_bytes=2*2**30):
    path=Path(path);payload=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    roots=BUDGET_ROOTS if budget_roots is None else budget_roots
    with SAVE_LOCK:
        # Reserve1MiB for bounded controller metadata/logs when using real study roots.
        limit=max_bytes-(2**20 if roots==BUDGET_ROOTS and roots else 0)
        if artifact_bytes(roots)+len(payload.encode())>limit:raise RuntimeError('artifact disk budget exceeded before write')
        with path.open('x') as stream:stream.write(payload)
        path.chmod(0o600)


def verify_hashes(mapping):
    for name,digest in mapping.items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=digest:
            raise ValueError('frozen file fingerprint mismatch')


def rss_bytes():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform=='darwin' else value*1024)


def validate_task(task):
    if (task['dataset'] not in REFERENCES or type(task['model_seed']) is not int or task['model_seed'] not in range(5)
        or task['algorithm']!=ALGORITHM or task['partition_seed']!=0
        or (task['T'],task['head_dimension'],task['full_dimension'],task['sigma_sum'])!=(100,33,2340,260)
        or task['reference_spread']!=REFERENCES[task['dataset']]
        or not 0<task['max_wall_minutes']<=60 or not 0<task['max_rss_gib']<=8):
        raise ValueError('task differs from frozen protocol')
    output=Path(task['output']).resolve()
    if not output.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('output outside research outputs')
    return output


def make_session(registry,spec,model_seed):
    from experiments.matched_pdp_frozen_backbone.research import Session
    return Session(registry,spec,model_seed=model_seed)


def evaluate(*args):
    from experiments.pdp_research_evaluation import evaluate as original
    return original(*args)


class Observer:
    """Only code-identity counts; random frame locals/returns are never read."""
    def __init__(self):self.counts=Counter()

    def __call__(self,frame,event,value):
        if event not in ('call','return'):return
        name=frame.f_code.co_name;filename=frame.f_code.co_filename.replace('\\','/')
        # Check random-code identity first, without reading any frame value.
        if '/matched_pdp_frozen_backbone/' in filename and filename.endswith(('/noise.py','/testing.py')):
            if name=='draw' and event=='return':self.counts['gaussian_vectors']+=1
            return
        if '/node_pdp/sampling.py' in filename:
            if name=='sample_once' and event=='return':self.counts['outer_sampling_calls']+=1
            return
        if '/experiments/' not in filename:return
        if event=='call' and ((filename.endswith('/release.py') and name=='export')
            or (filename.endswith('/node_dp_slots/accounting.py') and name=='charge')
            or (filename.endswith('/node_pdp/mechanism.py') and name=='reserve')):
            raise RuntimeError('production release and Ledger are forbidden')
        if event!='return':return
        if filename.endswith('/matched_pdp_frozen_backbone/kernel.py'):
            label={'execute':'training_calls','center':'bounded_queries','double_clip':'double_clips','step':'noisy_updates'}.get(name)
            if label:self.counts[label]+=1
        if filename.endswith('/node_dp_slots/adapter.py'):
            if name=='batch':self.counts['four_slot_batches']+=1
            if frame.f_code.co_qualname=='SlotCache.__init__':self.counts['slot_caches']+=1

    def __enter__(self):
        if sys.getprofile() is not None:raise RuntimeError('existing profiler')
        sys.setprofile(self);return self

    def __exit__(self,*args):sys.setprofile(None)

    def verify(self):
        expected=dict(training_calls=1,outer_sampling_calls=1,slot_caches=4,
                      gaussian_vectors=100,bounded_queries=100,double_clips=100,noisy_updates=100,four_slot_batches=100)
        if dict(self.counts)!=expected:raise RuntimeError('actual execution counts disagree with frozen mechanism')
        return dict(self.counts)



def abort_resource(private,start):
    """Failure marker is best effort; inability to write must never disable stop."""
    try:
        path=private/'failure.json'
        if not path.exists():save(path,dict(status='FAILED_NO_RETRY',exception_type='ResourceLimitOrMonitorFailure',
            worker_wall_seconds=time.monotonic()-start,peak_rss_bytes=rss_bytes()))
    except BaseException:
        pass
    finally:
        os._exit(70)


def run(task):
    global BUDGET_ROOTS
    start=time.monotonic();output=validate_task(task)
    verify_hashes(task['source_sha256']);verify_hashes(task['input_sha256'])
    output.mkdir(parents=True,exist_ok=False,mode=0o700);output.chmod(0o700)
    private=output/'restricted';private.mkdir(mode=0o700)
    BUDGET_ROOTS=tuple(Path(p) for p in task.get('artifact_budget_roots',[str(output)]))
    save(private/'run_started.json',dict(task_id=task['task_id'],algorithm_version=ALGORITHM,dataset=task['dataset'],
        model_seed=task['model_seed'],status='RUNNING',real_training_attempts=1,no_automatic_retry=True))
    stop=threading.Event()
    def watch():
        while not stop.wait(.25):
            try:
                violation=resource_violation(task,time.monotonic()-start,rss_bytes(),artifact_bytes(BUDGET_ROOTS))
            except BaseException:
                abort_resource(private,start)
            else:
                if violation:abort_resource(private,start)
    monitor=threading.Thread(target=watch,daemon=True);monitor.start()
    try:
        import torch
        torch.set_num_threads(1)
        from experiments.matched_pdp_frozen_backbone.config import Spec
        from experiments.node_sampling_utility import state_hash
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import mechanism as prod
        reg,domain,train,test,message,test_ids,_,evidence=load_inputs(task['dataset'],Path(task['registry']),REFERENCES[task['dataset']])
        if reg.fingerprint!=task['registry_fingerprint'] or not evidence['train_test_disjoint']:
            raise ValueError('registry or split isolation mismatch')
        spec=Spec(reg.universe,domain.directed)
        session=make_session(reg,spec,model_seed=task['model_seed'])
        initial=state_hash(session.model)
        if initial!=task['initial_model_sha256']:raise ValueError('public initialization fingerprint mismatch')
        if sum(p.numel() for p in session.model.parameters() if p.requires_grad)!=33:
            raise ValueError('training parameter count mismatch')
        guard,ledger=prod._PROCESS_GUARD,base._PROCESS_LEDGER;before=(guard._used,ledger.models)
        save(private/'configuration.json',dict(task_id=task['task_id'],dataset=task['dataset'],algorithm_version=ALGORITHM,
            model_seed=task['model_seed'],partition_seed=0,base=spec.public(),source_sha256=task['source_sha256'],
            input_sha256=task['input_sha256'],registry_fingerprint=reg.fingerprint,
            evaluation=dict(ic_p=1,steps=1,k=50,celf_reference=REFERENCES[task['dataset']]),
            publication_allowed=False,initial_model_sha256=initial))
        save(private/'input_checks.json',evidence)
        began=time.monotonic()
        with Observer() as observer:result=session.run(train)
        training_seconds=time.monotonic()-began;counts=observer.verify()
        if not all(math.isfinite(v) for v in result.parameters):raise RuntimeError('nonfinite final parameters')
        if (result.diagnostics['updates']!=100 or result.diagnostics['noise_coordinates']!=3300
            or not result.diagnostics['frozen_backbone_unchanged']
            or not result.diagnostics['all_projected_weights_nonnegative']):raise RuntimeError('Session completion mismatch')
        began=time.monotonic();final=evaluate(session.model,test,message,REFERENCES[task['dataset']]);evaluation_seconds=time.monotonic()-began
        if not math.isfinite(final['coverage']) or final['coverage']!=100*final['spread']/REFERENCES[task['dataset']]:
            if not math.isclose(final['coverage'],100*final['spread']/REFERENCES[task['dataset']],rel_tol=0.,abs_tol=1e-10):
                raise RuntimeError('evaluation arithmetic mismatch')
        final['topk_original_ids']=[test_ids[i] for i in final['topk_local_ids']]
        if guard is not prod._PROCESS_GUARD or ledger is not base._PROCESS_LEDGER or before!=(guard._used,ledger.models):
            raise RuntimeError('production controls changed')
        verify_hashes(task['source_sha256']);verify_hashes(task['input_sha256'])
        if resource_violation(task,time.monotonic()-start,rss_bytes(),artifact_bytes(BUDGET_ROOTS)):
            raise RuntimeError('resource limit')
        save(private/'training_diagnostics.json',dict(result.diagnostics,actual_counts=counts,random_frame_values_inspected=False,added_forward_or_backward_calls=0))
        save(private/'model.json',dict(parameters=list(result.parameters),configuration=result.configuration,publication_allowed=False))
        save(private/'final_evaluation.json',final)
        done=dict(status='COMPLETED',task_id=task['task_id'],dataset=task['dataset'],algorithm_version=ALGORITHM,
            model_seed=task['model_seed'],real_training_attempts=1,updates=100,noise_coordinates=3300,actual_counts=counts,
            frozen_backbone_unchanged=True,all_projected_weights_nonnegative=True,production_controls_unchanged=True,
            n=1,std=None,training_seconds=training_seconds,evaluation_seconds=evaluation_seconds,
            worker_wall_seconds=time.monotonic()-start,peak_rss_bytes=rss_bytes(),
            initial_model_sha256=initial,final_model_sha256=state_hash(session.model),
            account=result.configuration['account'],machine_pdp_certified=False,publication_allowed=False)
        save(private/'completion.json',done)
        return done
    except BaseException as error:
        path=private/'failure.json'
        if not path.exists():save(path,dict(status='FAILED_NO_RETRY',exception_type=type(error).__name__,
            worker_wall_seconds=time.monotonic()-start,peak_rss_bytes=rss_bytes()))
        raise
    finally:
        stop.set();monitor.join(timeout=1.);BUDGET_ROOTS=()


def main():
    os.umask(0o077)
    task=json.loads(Path(sys.argv[1]).read_text())
    if not task['source_sha256'] or not task['input_sha256']:
        raise ValueError('real worker requires nonempty frozen source/input manifests')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try:
        with open(os.devnull,'w') as sink,redirect_stdout(sink),redirect_stderr(sink):run(task)
    except BaseException as error:
        print(json.dumps(dict(status='FAILED_NO_RETRY',exception_type=type(error).__name__)))
        raise SystemExit(1)
    print(json.dumps(dict(status='COMPLETED',task_id=task['task_id'])))


if __name__=='__main__':main()

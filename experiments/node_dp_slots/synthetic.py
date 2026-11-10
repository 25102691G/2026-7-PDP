"""Synthetic-only CLI. No graph path/dataset option and no prepared-data loader."""
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import numpy as np
import torch
from .config import Spec
from .adapter import Graph
from .accounting import epsilon,minimum_scale,Ledger
from .mechanism import Engine
from .release import export


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def run(output):
    output=Path(output)
    if output.exists():raise FileExistsError('new run-id required')
    root=Path(__file__).resolve().parents[2]
    protected_files=list((root/'model').rglob('*.py'))+[root/'AGENTS.md']
    protected_files += [root/'experiments'/name for name in ('privacy.py','train.py','train_models.py','sampling.py',
                       'stable_slot_structure.py','stable_slot_gradients.py','stable_slot_training.py')]
    for relative in ('outputs/structure_sampling/stable-slots-l2h2-20261031-v1',
                     'outputs/gradient_diagnostics/email-stable-slots-clip3-20261031-v1',
                     'outputs/utility_sampling/email-stable-slots-partition0-zero-20261031-v1'):
        protected_files.extend(p for p in (root/relative).rglob('*') if p.is_file())
    protected={str(p):sha(p) for p in protected_files}
    spec=Spec(tuple(range(128)),directed=True)  # public synthetic U, before G
    output.mkdir(parents=True,mode=0o755)
    save(output/'public_configuration.json',spec.public())
    restricted=output/'restricted';restricted.mkdir(mode=0o700)
    mask=os.umask(0o077)
    torch.set_num_threads(1)
    try:
        edges=tuple((i,(i+k)%96) for i in range(96) for k in range(1,13))
        graph=Graph(tuple(range(96)),edges,True)
        empty=Engine.testing(spec,noise_seed=42).run(Graph((),(),True))
        ledger=Ledger();engine=Engine.privacy(spec,ledger=ledger)
        result=engine.run(graph)
        try:export(result)
        except PermissionError:blocked=True
        else:raise AssertionError('strict export must remain blocked')
        idealized=export(result,permit_idealized=True)
        save(restricted/'idealized_synthetic_artifact.json',idealized)
        try:ledger.charge()
        except ValueError:composed_blocked=True
        else:raise AssertionError('second model cannot reuse first-model budget')
        if not all(sha(p)==h for p,h in protected.items()):raise AssertionError('historical artifact changed')
        source=inspect.getsource(random.SystemRandom.random)+inspect.getsource(random.Random.normalvariate)
        (restricted/'python_random_source.txt').write_text(source)
        report=dict(status='completed',scope='artificial graph and gradients only',real_prepared_training=False,
                    epsilon_actual_idealized=epsilon()[0],optimal_order=epsilon()[1],minimum_reference_std=minimum_scale(),
                    actual_std_sum=260.,std_mean=16.25,empty_test=empty.testing_diagnostics,
                    privacy_synthetic_updates=100,privacy_synthetic_noise_calls=engine._source.calls,
                    parameter_coordinates=len(idealized['parameters']),strict_export_blocked=blocked,
                    second_model_budget_blocked=composed_blocked,machine_dp_certified=False,
                    protected_sha256=protected,protected_unchanged=True,
                    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
                    completed_at=datetime.now(timezone.utc).isoformat())
        save(restricted/'synthetic_evidence.json',report)
        save(restricted/'source_sha256.json',{str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
        print(json.dumps({k:report[k] for k in ('status','scope','epsilon_actual_idealized','privacy_synthetic_noise_calls','strict_export_blocked','machine_dp_certified')}))
        return report
    finally:os.umask(mask)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)


if __name__=='__main__':main()

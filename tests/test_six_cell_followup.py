"""Manual followup safety checks; only artificial graphs and saved public constants."""
import importlib
import json
import tempfile
import unittest
from decimal import Decimal, localcontext
from pathlib import Path


def module(name):
    try:
        return importlib.import_module('experiments.six_cell_followup.'+name)
    except ModuleNotFoundError as exc:
        raise AssertionError('six-cell launcher not implemented') from exc


class PlanTests(unittest.TestCase):
    def test_exact_sparse_matrix_and_once_only_tasks(self):
        p=module('plan');tasks=p.matrix()
        self.assertEqual(len(tasks),144);self.assertEqual(len({t['id'] for t in tasks}),144)
        self.assertEqual(sum(t['T'] for t in tasks),31950)
        self.assertEqual(sum(t['T'] for t in tasks if t['method']!='nonprivate'),15450)
        self.assertEqual(len({(t['dataset'],t['method']) for t in tasks}),6)
        for pair,configs in p.GROUPS.items():
            group=[t for t in tasks if (t['dataset'],t['method'])==pair]
            self.assertEqual(len(group),24)
            for name in {t['candidate'] for t in group}:
                self.assertEqual(sorted(t['seed'] for t in group if t['candidate']==name),[0,1,2])
        wrong=dict(tasks[0],T=999)
        with self.assertRaises(ValueError):p.validate_task(wrong)

    def test_actual_binary_sigma_never_exceeds_reference_rdp(self):
        p=module('plan')
        with localcontext() as ctx:
            ctx.prec=100
            for m in ['uniform1','pdp_v2','uniform4']:
                for steps in [100,150,300]:
                    n=p.noise_plan(m,steps)
                    sigma=Decimal.from_float(n['sigma_sum'])
                    self.assertGreaterEqual(sigma*sigma,Decimal(p.SIGMA100[m])**2*Decimal(steps)/100)
                    self.assertGreaterEqual(Decimal.from_float(n['temporal_factor'])**2,Decimal(steps)/100)
        self.assertIsNone(p.noise_plan('nonprivate',1000)['sigma_sum'])
        with self.assertRaises(ValueError):p.noise_plan('uniform4',1000)

    def test_all_results_kept_and_minimum_sd_only_after_mean_gate(self):
        p=module('plan');tasks=p.matrix()
        rows=[dict(task=t,coverage=70.,spread=70.,celf=100.) for t in tasks]
        values={'p0':(80.,82.,84.),'p1':(84.,84.,84.),'p2':(96.,96.,96.)}
        for r in rows:
            t=r['task']
            if (t['dataset'],t['method'])==('lastfm','uniform1') and t['candidate'] in values:
                r['coverage']=r['spread']=values[t['candidate']][t['seed']]
        result=p.summarize(tasks,rows)
        self.assertTrue(result['selection_permitted'])
        winner=next(x for x in result['decisions'] if x['dataset']=='lastfm' and x['method']=='uniform1')
        self.assertEqual(winner['selected_candidate'],'p1')
        hepph=next(x for x in result['decisions'] if x['dataset']=='hepph' and x['method']=='nonprivate')
        self.assertEqual(hepph['status'],'TARGET_NOT_MET')
        partial=p.summarize(tasks,rows[:-1]);self.assertFalse(partial['selection_permitted'])
        self.assertNotIn('decisions',partial)
        one=p.summarize(tasks,rows[:1])['cells'][0];self.assertEqual(one['n_runs'],1)
        self.assertIsNone(one['sample_sd']);self.assertIsNone(one['sample_variance'])
        with self.assertRaises(ValueError):p.summarize(tasks,rows+[rows[0]])
        self.assertFalse(p.eligible('hepph','nonprivate',95.))
        self.assertFalse(p.eligible('hepph','pdp_v2',90.))
        self.assertTrue(p.eligible('facebook','uniform1',80.))
        self.assertTrue(p.eligible('facebook','uniform1',85.))


class KernelTests(unittest.TestCase):
    def test_extended_sampler_still_keeps_unaffected_slots_local(self):
        k=module('kernel')
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots.config import Spec,memberships
        ids=tuple(range(96));spec=Spec(ids,False)
        g=Graph(ids,tuple((i,j) for i in ids for j in ids if i<j),False)
        for n,m in [(50,3),(70,5),(50,7)]:
            a=k.CandidateAdapter(spec,g,n=n,M=m);b=k.CandidateAdapter(spec,g.delete(0),n=n,M=m)
            for j in set(range(4))-set(memberships(0)):
                self.assertEqual(a.fingerprints()[j],b.fingerprints()[j])
                self.assertEqual(a.batch()[j],b.batch()[j])

    def test_artificial_extended_loop_has_exact_budget_and_unchanged_update(self):
        k=module('kernel')
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots.config import Spec
        from experiments.node_dp_slots import mechanism as base
        from experiments.dataset_parameter_screen.kernel import run_private_synthetic as old
        import torch
        spec=Spec((0,),False);g=Graph((),(),False)
        a=base.new_model(0);b=base.new_model(0)
        k.run_private_synthetic(a,spec,g,n=40,M=4,steps=100,method='uniform4',noise_seed=7)
        old(b,spec,g,n=40,M=4,steps=100,method='uniform4',noise_seed=7)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))
        counts=k.run_private_synthetic(a,spec,g,n=50,M=7,steps=150,method='uniform1',noise_seed=8)
        self.assertEqual((counts['updates'],counts['gaussian_vectors'],counts['projections']),(150,150,150))


class ControllerTests(unittest.TestCase):
    def test_budget_persists_and_claimed_work_cannot_restart(self):
        r=module('run')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);task=module('plan').matrix()[0]
            r.claim(root,task['id'],task)
            with self.assertRaises(RuntimeError):r.pending_tasks(root,[task])
            r.save(root/'worker_budget.json',dict(consumed_seconds=123.,active_task=None))
            r.charge_worker_budget(root,7.)
            self.assertAlmostEqual(r.read(root/'worker_budget.json')['consumed_seconds'],130.)
            self.assertAlmostEqual(r.remaining_worker_budget(root,14400.),14270.)
            r.save(root/'worker_budget.json',dict(consumed_seconds=130.,active_task='interrupted'))
            with self.assertRaises(RuntimeError):r.remaining_worker_budget(root,14400.)

    def test_new_np_command_uses_original_trainer_for_exact_thousand_steps(self):
        r=module('run');p=module('plan')
        task=next(t for t in p.matrix() if t['method']=='nonprivate' and t['T']==1000)
        command=r.nonprivate_command(task,Path('/tmp/artificial-six-cell'))
        self.assertIn('--no-dp',command)
        self.assertEqual(command[command.index('--n_iteration')+1],'1000')
        self.assertNotIn('--auto-parameters',command)

if __name__=='__main__':unittest.main()

class ControllerIntegrationTests(unittest.TestCase):
    def test_timeout_consumes_one_claim_kills_worker_and_refuses_resume(self):
        import subprocess
        import sys
        from unittest.mock import patch
        r=module('run');p=module('plan');launched=[]
        real_popen=subprocess.Popen
        def artificial_worker(*args,**kwargs):
            # Actual isolated subprocess, deliberately sleeps; never imports a model.
            proc=real_popen([sys.executable,'-c','import time;time.sleep(10)'],**kwargs)
            launched.append(proc);return proc
        manifest=dict(tasks=p.matrix(),limits=dict(total_worker_wall_seconds=.15,task_wall_seconds=1,minimum_free_disk_bytes=0))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(r,'preflight',return_value=manifest),patch.object(r,'checked_manifest',return_value=manifest),patch.object(r.subprocess,'Popen',side_effect=artificial_worker):
                with self.assertRaises(subprocess.TimeoutExpired):r.run(root)
                self.assertEqual(len(launched),1);self.assertIsNotNone(launched[0].poll())
                status=r.read(root/'status.json')
                self.assertEqual((status['failed'],status['unexecuted'],status['retries']),(1,143,0))
                self.assertIsNone(r.read(root/'worker_budget.json')['active_task'])
                self.assertGreater(r.read(root/'worker_budget.json')['consumed_seconds'],0)
                with self.assertRaises(RuntimeError):r.run(root,resume=True)
                self.assertEqual(len(launched),1)

    def test_exhausted_budget_never_claims_another_task(self):
        from unittest.mock import patch
        r=module('run');p=module('plan')
        manifest=dict(tasks=p.matrix(),limits=dict(total_worker_wall_seconds=1,task_wall_seconds=1,minimum_free_disk_bytes=0))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            r.save(root/'worker_budget.json',dict(consumed_seconds=1.,active_task=None))
            with patch.object(r,'preflight',return_value=manifest),patch.object(r,'checked_manifest',return_value=manifest):
                with self.assertRaisesRegex(RuntimeError,'budget exhausted'):r.run(root)
            self.assertFalse((root/'models').exists())

class ReviewRegressionTests(unittest.TestCase):
    def test_read_only_status_does_not_create_or_rewrite_files(self):
        r=module('run');p=module('plan')
        self.assertTrue(hasattr(r,'status_readonly'),'status must have a separate read-only entry')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);before=list(root.iterdir())
            status=r.status_readonly(root,dict(tasks=p.matrix()))
            self.assertEqual(status['unexecuted'],144)
            self.assertEqual(list(root.iterdir()),before)

    def test_unmet_groups_report_pareto_tradeoffs(self):
        p=module('plan');tasks=p.matrix()
        rows=[dict(task=t,coverage=70.,spread=70.,celf=100.) for t in tasks]
        for r in rows:
            t=r['task']
            if (t['dataset'],t['method'])==('hepph','nonprivate'):
                vals=(88.,90.,92.) if t['candidate']=='p0' else (81.,81.,81.)
                r['coverage']=r['spread']=vals[t['seed']]
        result=p.summarize(tasks,rows)
        decision=next(d for d in result['decisions'] if (d['dataset'],d['method'])==('hepph','nonprivate'))
        self.assertEqual(decision['status'],'TARGET_NOT_MET')
        self.assertIn('p0',decision.get('pareto_candidates',[]))
        self.assertIn('p1',decision.get('pareto_candidates',[]))

"""Behavioral checks: no protected graph, secret random source or real fit."""
import json
import tempfile
import unittest
from pathlib import Path


class PlanTests(unittest.TestCase):
    def test_full_matrix_has_three_seeds_per_candidate_without_duplicate_fits(self):
        from experiments.dataset_parameter_screen.plan import matrix
        tasks = matrix()
        self.assertEqual(len(tasks), 480)
        self.assertEqual(len({t['id'] for t in tasks}), 480)
        for d in ('email', 'bitcoin', 'lastfm', 'hepph', 'facebook'):
            for m in ('nonprivate', 'uniform1', 'pdp_v2', 'uniform4'):
                cell = [t for t in tasks if t['dataset'] == d and t['method'] == m]
                self.assertEqual(len(cell), 24)
                self.assertEqual({t['seed'] for t in cell}, {0, 1, 2})
                configs = {(t['n'], t['M'], t['T']) for t in cell}
                self.assertEqual(len(configs), 8)
        self.assertEqual(sum(t['T'] for t in tasks), 96000)

    def test_noise_scaling_does_not_spend_more_rdp_for_long_training(self):
        from decimal import Decimal
        from experiments.dataset_parameter_screen.plan import noise_plan
        for m, sigma in [('uniform1', 486), ('pdp_v2', 260), ('uniform4', 139)]:
            for steps in (100, 300, 500):
                c = noise_plan(m, steps)
                factor = Decimal.from_float(c['temporal_factor'])
                self.assertGreaterEqual(factor * factor, Decimal(steps) / 100)
                self.assertAlmostEqual(c['sigma_sum'], sigma * (steps / 100) ** .5, places=10)
        self.assertIsNone(noise_plan('nonprivate', 500)['sigma_sum'])
        with self.assertRaises(ValueError): noise_plan('uniform1', 101)

    def test_claims_never_overwrite_or_retry_failed_and_inflight_tasks(self):
        from experiments.dataset_parameter_screen.run import claim, pending_tasks
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claim(root, 'example')
            with self.assertRaises(FileExistsError): claim(root, 'example')
            with self.assertRaises(RuntimeError): pending_tasks(root, [{'id': 'example'}])
            (root / 'models/example/failure.json').write_text('{}')
            with self.assertRaises(RuntimeError): pending_tasks(root, [{'id': 'example'}])

    def test_summary_refuses_incomplete_cells_and_does_not_hide_bad_seed(self):
        from experiments.dataset_parameter_screen.plan import summarize, matrix
        tasks = matrix()
        rows = [dict(task=t, coverage=80., spread=80., celf=100.) for t in tasks]
        # One poor run must survive in its cell, lowering the mean and increasing SD.
        rows[0].update(coverage=20., spread=20.)
        summary = summarize(tasks, rows)
        cell = next(c for c in summary['cells'] if c['dataset']=='email' and c['method']=='nonprivate' and c['candidate']=='baseline')
        self.assertEqual(cell['n_runs'], 3)
        self.assertEqual(cell['mean'], 60.)
        self.assertAlmostEqual(cell['sample_sd'], 34.64101615137755)
        self.assertEqual(cell['below50'], 1)
        self.assertEqual(cell['minimum'], 20.)
        self.assertEqual(summarize(tasks, rows[:-1])['status'], 'INCOMPLETE')
        with self.assertRaises(ValueError): summarize(tasks, rows + [rows[0]])

    def test_mean_gain_alone_is_not_recommended_when_sd_worsens(self):
        from experiments.dataset_parameter_screen.plan import summarize, matrix
        tasks = matrix()
        rows = [dict(task=t, coverage=80., spread=80., celf=100.) for t in tasks]
        for r in rows:
            if r['task']['candidate']=='steps300':
                r['coverage'] = r['spread'] = (80., 90., 100.)[r['task']['seed']]
        report = summarize(tasks, rows)
        self.assertFalse(any(c['candidate']=='steps300' and c['passes_baseline_guards'] for c in report['cells']))


if __name__ == '__main__': unittest.main()

class KernelTests(unittest.TestCase):
    def fixture(self):
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots.config import Spec
        ids=tuple(range(96))
        return Spec(ids,False),Graph(ids,tuple((u,v) for u in ids for v in ids if u<v),False)

    def test_baseline_adapter_matches_original_and_keeps_unaffected_slots_local(self):
        from experiments.dataset_parameter_screen.kernel import CandidateAdapter
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_dp_slots.config import memberships
        spec,g=self.fixture()
        old=Adapter(spec,g);new=CandidateAdapter(spec,g,n=40,M=4)
        self.assertEqual(old.fingerprints(),new.fingerprints())
        self.assertEqual(old.batch(),new.batch())
        for n,m in [(20,2),(60,6),(80,8)]:
            a=CandidateAdapter(spec,g,n=n,M=m)
            b=CandidateAdapter(spec,g.delete(0),n=n,M=m)
            for j in set(range(4))-set(memberships(0)):
                self.assertEqual(a.fingerprints()[j],b.fingerprints()[j])
                for _ in range(2): self.assertEqual(a.batch()[j],b.batch()[j])

    def test_frozen_update_and_projection_are_identical_under_baseline_candidate(self):
        import torch
        from experiments.dataset_parameter_screen.kernel import run_private_synthetic
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.noise import TestGaussian
        from experiments.node_dp_slots.training import execute
        from experiments.node_pdp_v2_projected import policy
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots.config import Spec
        # Tiny empty artificial graph exercises all100 steps without expensive forwards.
        spec=Spec(tuple(range(3)),False);g=Graph((),(),False)
        a=base.new_model(1);b=base.new_model(1);layout=policy.Layout.from_model(a)
        values,_,_=execute(spec,g,TestGaussian(7),model=a,step=lambda m,s,z:policy.step(m,s,z,layout))
        counts=run_private_synthetic(b,spec,g,n=40,M=4,steps=100,method='pdp_v2',noise_seed=7)
        self.assertEqual(counts['updates'],100)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))

    def test_new_loop_reuses_center_on_nonempty_artificial_batches(self):
        import torch
        from experiments.dataset_parameter_screen.kernel import CandidateAdapter, run_private_synthetic
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.noise import TestGaussian
        from experiments.node_pdp_v2_projected import policy
        spec,g=self.fixture();adapter=CandidateAdapter(spec,g,n=40,M=4)
        a=base.new_model(0);b=base.new_model(0);layout=policy.Layout.from_model(a)
        source=TestGaussian(9)
        for _ in range(2):
            signal,_=base.center(a,adapter.batch());policy.step(a,signal,source,layout)
        run_private_synthetic(b,spec,g,n=40,M=4,steps=2,method='pdp_v2',noise_seed=9,reference_steps=100)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))

class SafetyTests(unittest.TestCase):
    def test_synthetic_entry_rejects_missing_seed_before_secret_source_creation(self):
        from experiments.dataset_parameter_screen.kernel import run_private_synthetic
        with self.assertRaisesRegex(ValueError, 'synthetic seed'):
            run_private_synthetic(None,None,None,n=40,M=4,steps=100,method='pdp_v2',noise_seed=None)

    def test_original_nonprivate_command_carries_candidate_and_no_auto_overrides(self):
        from experiments.dataset_parameter_screen.plan import matrix
        from experiments.dataset_parameter_screen.run import nonprivate_command
        t=next(t for t in matrix() if t['dataset']=='hepph' and t['method']=='nonprivate' and t['candidate']=='joint')
        args=nonprivate_command(t,Path('/tmp/artificial-command-test'))
        self.assertEqual(args[args.index('--n_iteration')+1],'300')
        self.assertEqual(args[args.index('--subgraph_size')+1],'60')
        self.assertEqual(args[args.index('--score_upper_bound')+1],'2')
        self.assertIn('--no-dp',args)
        self.assertNotIn('--auto-parameters',args)
        self.assertNotIn('--clip-non-private',args)

class OriginalEntryTests(unittest.TestCase):
    def test_original_entry_inherits_thread_limit_and_restores_argv(self):
        import sys
        import torch
        from experiments.dataset_parameter_screen.run import invoke_original
        with tempfile.TemporaryDirectory() as tmp:
            script=Path(tmp)/'artificial.py';result=Path(tmp)/'result.json'
            script.write_text("import torch,json,sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text(json.dumps({'threads':torch.get_num_threads(),'args':sys.argv[2:]}))\n")
            previous=sys.argv[:];threads=torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                invoke_original([sys.executable,str(script),str(result),'literal-argument'])
                self.assertEqual(json.loads(result.read_text()),dict(threads=1,args=['literal-argument']))
                self.assertEqual(sys.argv,previous)
            finally:torch.set_num_threads(threads)

class PartialStatisticsTests(unittest.TestCase):
    def test_partial_statistics_show_actual_n_and_na_sd_without_selection(self):
        from experiments.dataset_parameter_screen.plan import matrix,summarize
        tasks=matrix();report=summarize(tasks,[dict(task=tasks[0],coverage=40.,spread=40.,celf=100.)])
        self.assertFalse(report['selection_permitted'])
        cell=next(c for c in report['cells'] if c['dataset']=='email' and c['method']=='nonprivate' and c['candidate']=='baseline')
        self.assertEqual(cell['n_runs'],1)
        self.assertEqual(cell['mean'],40.)
        self.assertIsNone(cell['sample_sd'])
        self.assertNotIn('pareto_mean_sd',cell)

class ExtendedBudgetTests(unittest.TestCase):
    def test_long_candidate_executes_exactly_five_hundred_updates_and_draws(self):
        from experiments.dataset_parameter_screen.kernel import run_private_synthetic
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.config import Spec
        from experiments.node_dp_slots.adapter import Graph
        model=base.new_model(2)
        counts=run_private_synthetic(model,Spec((0,),False),Graph((),(),False),n=80,M=6,steps=500,method='uniform1',noise_seed=22)
        self.assertEqual((counts['updates'],counts['gaussian_vectors'],counts['projections']),(500,500,500))
        self.assertEqual(counts['gaussian_coordinates'],1170000)

class ConcurrentStatusTests(unittest.TestCase):
    def test_atomic_status_writes_do_not_share_a_temporary_filename(self):
        from concurrent.futures import ThreadPoolExecutor
        from experiments.dataset_parameter_screen.run import save
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'status.json'
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda i:save(target,dict(counter=i)),range(80)))
            self.assertIn(json.loads(target.read_text())['counter'],range(80))

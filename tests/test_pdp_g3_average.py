"""Artificial-only tests for the fixed post-update parameter mean."""
import unittest
from unittest.mock import patch


class G3Tests(unittest.TestCase):
    def api(self):
        from experiments import pdp_g3_average
        return pdp_g3_average

    def test_exact25_trajectories_not50(self):
        cells=self.api().task_matrix()
        self.assertEqual(len(cells),25)
        self.assertEqual({x['model_seed'] for x in cells},set(range(5)))
        self.assertEqual({x['dataset'] for x in cells},{'email','bitcoin','lastfm','hepph','facebook'})

    def test_only_Finder_metadata_excluded_not_research_or_dataset_metadata(self):
        import hashlib,tempfile
        from pathlib import Path
        from experiments.run_pdp_g3 import historical_diff
        with tempfile.TemporaryDirectory() as tmp:
            paths=[Path(tmp)/name for name in ('.DS_Store','metadata.json','result.json')]
            for p in paths:p.write_bytes(b'old')
            expected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            paths[0].write_bytes(b'Finder metadata change')
            diff=historical_diff(expected)
            self.assertEqual(diff['research_mismatches'],[])
            self.assertEqual(len(diff['platform_metadata_mismatches']),1)
            for p in paths[1:]:
                p.write_bytes(b'changed scientific contents')
                self.assertIn(str(p),[x['path'] for x in historical_diff(expected)['research_mismatches']])

    def test_constant_path_and_initial_state_excluded(self):
        import torch
        api=self.api();model=torch.nn.Module();model.register_parameter('p',torch.nn.Parameter(torch.full((2340,),-999.)))
        accumulator=api.ParameterMean(model)
        for t in range(1,101):
            with torch.no_grad():model.p.fill_(t)
            accumulator.observe(model)
        self.assertTrue(torch.equal(accumulator.finish(),torch.full((2340,),50.5)))
        with self.assertRaises(RuntimeError):accumulator.observe(model)
        constant=api.ParameterMean(model)
        for _ in range(100):constant.observe(model)
        self.assertTrue(torch.equal(constant.finish(),model.p.detach()))

    def test_incomplete_mean_refused(self):
        import torch
        model=torch.nn.Module();model.register_parameter('p',torch.nn.Parameter(torch.zeros(2340)))
        a=self.api().ParameterMean(model)
        with self.assertRaises(RuntimeError):a.finish()

    def test_test_sources_cannot_enter_real_candidate(self):
        from tests.test_pdp_g2_controlled import G2Tests
        r,s,g=G2Tests().fixture();cls=self.api().AverageSession
        with self.assertRaises(ValueError):cls(r,s,'node_pdp_v1')
        session=cls.synthetic(r,s,'node_pdp_v2_projected',sampling_seed=1,noise_seed=2)
        session.purpose='research'
        from experiments.node_pdp import sampling
        with patch.object(sampling,'sample_once',side_effect=AssertionError('must refuse before sampling')):
            with self.assertRaises(PermissionError):session.run(g)

    def test_artificial_full_path_mean_and_original_trajectory_identical(self):
        import torch
        torch.set_num_threads(1)
        api=self.api()
        from tests.test_pdp_g2_controlled import G2Tests
        r,s,g=G2Tests().fixture()
        from experiments.node_pdp_research import Session
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_dp_slots.accounting import Ledger
        from experiments.node_pdp import sampling
        Model=base.runtime()[0];old_forward=Model.forward;old_backward=torch.Tensor.backward
        old_init=Adapter.__init__;old_batch=Adapter.batch
        runs=[];reference_path=[];original_observe=api.ParameterMean.observe
        def capture_mean(a,m):
            reference_path.append(torch.cat([p.detach().flatten().double() for p in m.parameters()]).clone())
            return original_observe(a,m)
        def run(observed):
            counts=dict(forward=0,backward=0);tapes=[];caches=[]
            def forward(m,*args,**kw):counts['forward']+=1;return old_forward(m,*args,**kw)
            def backward(t,*args,**kw):counts['backward']+=1;return old_backward(t,*args,**kw)
            def init(a,*args,**kw):old_init(a,*args,**kw);caches.append(a.fingerprints())
            def batch(a):
                result=old_batch(a);tapes.append((tuple(tuple(x) for x in result),tuple(c.batch_rng.getstate() for c in a.caches)))
                return result
            cls=api.AverageSession if observed else Session
            import random
            public_random_before=random.getstate();torch_random_before=torch.random.get_rng_state().clone()
            with patch.object(Model,'forward',forward),patch.object(torch.Tensor,'backward',backward),patch.object(Adapter,'__init__',init),patch.object(Adapter,'batch',batch):
                with patch.object(sampling,'sample_once',wraps=sampling.sample_once) as bern:
                    session=cls.synthetic(r,s,'node_pdp_v2_projected',sampling_seed=17,noise_seed=19)
                    result=session.run(g)
                    self.assertEqual(bern.call_count,1)
            self.assertEqual(random.getstate(),public_random_before)
            self.assertTrue(torch.equal(torch.random.get_rng_state(),torch_random_before))
            runs.append((counts,tapes,caches,session))
            return result
        with patch.object(Ledger,'charge',side_effect=AssertionError('no production ledger')):
            before=run(False)
            with patch.object(api.ParameterMean,'observe',capture_mean):after=run(True)
        self.assertEqual(before.parameters,after.last.parameters)
        self.assertEqual(runs[0][:3],runs[1][:3])
        self.assertGreater(runs[0][0]['forward'],0)
        self.assertEqual(len(reference_path),100)
        expected=torch.stack(reference_path).sum(0)/100
        actual=torch.tensor(after.average_parameters,dtype=torch.float32)
        # Independent stack reduction can differ at FP64 rounding; compare FP32 outputs.
        self.assertTrue(torch.equal(actual,expected.float()))
        self.assertEqual(after.last.diagnostics['noise_calls'],100)
        self.assertEqual(after.last.diagnostics['outer_sampling_calls'],1)
        self.assertTrue(after.checks['last_unchanged'])
        self.assertTrue(after.checks['average_head_nonnegative'])
        self.assertTrue(all(p.grad is None for p in runs[1][3].average_model.parameters()))


if __name__=='__main__':unittest.main()

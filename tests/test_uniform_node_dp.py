"""Artificial graphs only. No prepared data or real research training."""
import math
import random
import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch
import torch


class UniformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self):
        from experiments.uniform_node_dp.config import Spec
        from experiments.node_dp_slots.adapter import Graph
        ids=tuple(range(24))
        return Spec(ids),Graph(ids,tuple((v,(v+1)%24) for v in ids),True)

    def test_frozen_separate_configuration(self):
        from experiments.node_dp_slots.config import Spec as PDP
        from experiments.uniform_node_dp.config import Spec, validate
        s,g=self.fixture();validate(s)
        self.assertEqual((s.B,s.C,s.K,s.T,s.dimension,s.s_sum,s.lr),(16,3.,2,100,2340,486.,.005))
        self.assertEqual(s.public()['epsilon_target'],1.)
        self.assertFalse(s.public()['outer_bernoulli'])
        self.assertEqual(PDP(s.universe).s_sum,260.)
        with self.assertRaises(TypeError):validate(PDP(s.universe))
        from experiments.node_pdp.registry import Registry
        from experiments.node_pdp.accounting import validate_public
        with self.assertRaises(TypeError):validate_public(Registry(s.universe,tuple((v,2) for v in s.universe),source='artificial'),s)
        with self.assertRaises(Exception):s.s_sum=260.

    def test_noise_scale_and_all_coordinates(self):
        from experiments.uniform_node_dp.noise import SecretGaussian,TestGaussian
        calls=[]
        with patch.object(random.SystemRandom,'normalvariate',side_effect=lambda m,s:(calls.append((m,s)) or 0.)):
            z=SecretGaussian();v=z.draw()
        self.assertEqual(v.shape,(2340,));self.assertEqual(v.dtype,torch.float64)
        self.assertEqual(calls,[(0.,486.)]*2340);self.assertEqual(z.calls,1)
        a=TestGaussian(9).draw();rng=random.Random(9)
        self.assertEqual(a.tolist(),[rng.normalvariate(0.,486.) for _ in range(2340)])

    def test_full_loop_matches_manual_reference_and_no_bernoulli(self):
        from experiments.uniform_node_dp.research import Session
        from experiments.uniform_node_dp.noise import TestGaussian
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_pdp_v2_projected import policy
        s,g=self.fixture();m=base.new_model(3);layout=policy.Layout.from_model(m)
        initial=layout.target(m).detach().clone();a=Adapter(s,g);noise=TestGaussian(7)
        for _ in range(100):
            signal,slots=base.center(m,a.batch())
            self.assertTrue(all(float(x.norm())<=3 for x in slots))
            policy.step(m,signal,noise,layout)
        expected=tuple(float(v) for v in torch.cat([p.detach().reshape(-1) for p in m.parameters()]))
        with patch('experiments.node_pdp.sampling.sample_once',side_effect=AssertionError('outer Bernoulli forbidden')):
            session=Session.synthetic(s,noise_seed=7,model_seed=3)
            self.assertTrue(torch.equal(session.layout.target(session.model),initial))
            r=session.run(g)
        self.assertEqual(r.parameters,expected)
        self.assertEqual(r.diagnostics['noise_calls'],100)
        self.assertEqual(r.diagnostics['noise_coordinates'],234000)
        self.assertEqual(r.diagnostics['projection_calls'],100)
        self.assertEqual(r.diagnostics['outer_sampling_calls'],0)
        self.assertTrue(r.diagnostics['all_projected_weights_nonnegative'])
        with self.assertRaises(PermissionError):session.run(g)

    def test_four_slots_membership_and_delete_locality(self):
        from experiments.node_dp_slots.config import memberships
        from experiments.node_dp_slots.adapter import Adapter
        s,g=self.fixture();a=Adapter(s,g);b=Adapter(s,g.delete(4))
        self.assertEqual(len(a.caches),4)
        for v in s.universe:self.assertEqual(len(set(memberships(v))),2)
        self.assertEqual([len(x) for x in a.batch()],[4]*4)
        for j in set(range(4))-set(memberships(4)):
            # Recreate before batch so public batch cursors agree.
            self.assertEqual(Adapter(s,g).fingerprints()[j],b.fingerprints()[j])

    def test_double_clipping_and_twelve_sensitivity_bound(self):
        from experiments.node_dp_slots.aggregation import clip,double_clip
        raw=torch.full((4,4,2340),1e6,dtype=torch.float64)
        x=torch.stack([torch.stack([clip(v) for v in row]) for row in raw])
        signal,slots=double_clip(x)
        self.assertTrue(all(float(v.norm())<=3 for row in x for v in row))
        self.assertTrue(all(float(v.norm())<=3 for v in slots))
        other=slots.clone();other[:2]=-other[:2]
        self.assertLessEqual(float((signal-other.sum(0)).norm()),12.)
        self.assertLessEqual(float(signal.norm()),12.)

    def test_accountant_independent_decimal_equations(self):
        from experiments.uniform_node_dp.accounting import account
        with localcontext() as c:
            c.prec=50;values=[]
            for i in range(11,1000):
                a=Decimal(i)/10;rho=100*a*144/(2*Decimal(486)**2)
                e=rho-((Decimal('1e-5')).ln()+a.ln())/(a-1)+((a-1)/a).ln()
                values.append((e,a))
            e,a=min(values)
        with patch('experiments.node_dp_slots.accounting.epsilon',side_effect=AssertionError('independent accountant')):
            r=account()
        self.assertAlmostEqual(r['epsilon'],float(e),places=13)
        self.assertEqual(r['rdp_order'],float(a));self.assertLess(r['epsilon'],1)
        self.assertAlmostEqual(r['epsilon'],.998692964401014,places=10)
        self.assertEqual(r['delta'],1e-5)

    def test_empty_graph_still_100_steps_and_controls_unchanged(self):
        from experiments.uniform_node_dp.research import Session
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import mechanism as pdp
        s,g=self.fixture();ledger,guard=base._PROCESS_LEDGER,pdp._PROCESS_GUARD
        before=(ledger.models,guard._used)
        r=Session.synthetic(s,noise_seed=8).run(Graph((),(),True))
        self.assertEqual(r.diagnostics['updates'],100)
        self.assertEqual((ledger.models,guard._used),before)
        self.assertIs(ledger,base._PROCESS_LEDGER);self.assertIs(guard,pdp._PROCESS_GUARD)
        from experiments.node_pdp_v2_projected import release
        with self.assertRaises(PermissionError):release.export(r,permit_idealized=True)

    def test_real_session_rejects_test_source_and_wrong_domain(self):
        from experiments.uniform_node_dp.research import Session
        from experiments.uniform_node_dp.noise import TestGaussian
        from experiments.node_dp_slots.adapter import Graph
        s,g=self.fixture();session=Session(s);session._noise=TestGaussian(0)
        with self.assertRaises(PermissionError):session.run(g)
        with self.assertRaises(ValueError):Session.synthetic(s,noise_seed=0).run(Graph((100,),(),True))

    def test_projection_changes_only_32_coordinates(self):
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp_v2_projected import policy
        from experiments.uniform_node_dp.noise import TestGaussian
        a=base.new_model(4);b=base.new_model(4);layout=policy.Layout.from_model(a)
        signal=torch.zeros(2340,dtype=torch.float64)
        base.update(a.parameters(),signal,TestGaussian(3))
        policy.step(b,signal,TestGaussian(3),layout)
        av=torch.cat([p.detach().reshape(-1) for p in a.parameters()]);bv=torch.cat([p.detach().reshape(-1) for p in b.parameters()])
        self.assertTrue(torch.equal(av[:layout.start],bv[:layout.start]))
        self.assertTrue(torch.equal(av[layout.stop:],bv[layout.stop:]))
        self.assertTrue(torch.equal(av[layout.start:layout.stop].clamp_min(0),bv[layout.start:layout.stop]))

    def test_cli_artificial_input_and_restricted_records(self):
        import argparse
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from experiments import run_uniform_node_dp_research as cli
        from experiments.uniform_node_dp.research import Session
        from experiments.node_dp_slots.adapter import Graph
        s,g=self.fixture()
        def synthetic(spec,model_seed):return Session.synthetic(spec,noise_seed=7,model_seed=model_seed)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);output=root/'outputs'/'artificial'
            args=argparse.Namespace(dataset='email',registry=root/'unused.json',model_seed=2,output=output)
            inputs=(SimpleNamespace(universe=s.universe,fingerprint='artificial'),s,Graph((),(),True),None,None,tuple(range(50)),{},dict(domain_scope='artificial'))
            with patch.object(cli,'ROOT',root),patch.object(cli,'load_inputs',return_value=inputs),patch.object(cli,'Session',side_effect=synthetic),patch.object(cli,'evaluate',return_value=dict(spread=50,coverage=100*50/432,topk_local_ids=list(range(50)))):
                cli.run(args)
                with self.assertRaises(FileExistsError):cli.run(args)
            cfg=json.loads((output/'restricted/configuration.json').read_text())
            done=json.loads((output/'restricted/completion.json').read_text())
            self.assertEqual(cfg['base']['s_sum'],486.)
            self.assertEqual(cfg['model_seed'],2);self.assertFalse(cfg['publication_allowed'])
            self.assertEqual(done['updates'],100);self.assertLess(done['account']['epsilon'],1)
            for f in (output/'restricted').iterdir():
                self.assertEqual(f.stat().st_mode & 0o777,0o600)
                text=f.read_text()
                for forbidden in ['noise_seed','sampling_mask','random_bytes','noise_vector']:
                    self.assertNotIn(forbidden,text)


if __name__=='__main__':unittest.main()

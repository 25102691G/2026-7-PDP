"""C2 artificial graphs only; no prepared loaders/checkpoints. FP32 tolerances frozen."""
import importlib.util
import math
import random
import unittest
from unittest.mock import patch
import torch

RTOL=2e-5
ATOL=2e-5


class RowSumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.rowsum_pdp'), 'independent candidate package required')
        from experiments.rowsum_pdp.model import Model,Layout
        return Model,Layout

    def message(self,n=5,edges=((0,1),(0,2),(1,2),(2,3)),directed=True):
        from experiments.node_dp_slots.mechanism import runtime
        ig=runtime()[4];g=ig.Graph(n=n,edges=edges,directed=directed)
        return runtime()[2](g,'ones',32,directed,False)

    def fixture(self):
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_pdp.registry import Registry
        from experiments.rowsum_pdp.config import Spec
        ids=tuple(range(128));reg=Registry(ids,tuple((v,2) for v in ids),source='public-artificial-C2')
        g=Graph(tuple(range(96)),tuple((v,(v+k)%96) for v in range(96) for k in range(1,13)),True)
        return reg,Spec(ids),g

    def test_package_exists(self):self.api()

    def test_initial_reconstruction_every_coordinate_all_seeds(self):
        Model,Layout=self.api()
        from experiments.node_dp_slots.mechanism import new_model
        for seed in range(5):
            m=Model(seed);state=m.reconstruct();old=new_model(seed).state_dict()
            self.assertEqual(list(state),list(old))
            for name in old:self.assertTrue(torch.equal(old[name],state[name]),name)
            self.assertEqual(sum(p.numel() for p in m.parameters()),1283)
            layout=Layout.from_model(m);self.assertEqual((layout.total,layout.start,layout.stop),(1283,1185,1217))
            self.assertTrue(any(bool((m.layer3.fc.weight<0).any()) for _ in [0]))

    def test_forward_equivalence_directed_undirected_isolated_empty(self):
        Model,_=self.api()
        from experiments.node_dp_slots.mechanism import new_model
        cases=[(5,((0,1),(0,2),(1,2),(2,3)),True),(5,((0,1),(1,2),(2,3)),False),(4,(),True),(4,(),False),(0,(),True)]
        for seed in range(5):
            model=Model(seed);old=new_model(seed)
            for n,edges,directed in cases:
                msg=self.message(n,edges,directed);x=msg.ndata['feat']
                torch.testing.assert_close(model.logits(msg,x),old.logits(msg,x),rtol=RTOL,atol=ATOL)
                torch.testing.assert_close(model(msg,x),old(msg,x),rtol=RTOL,atol=ATOL)

    def test_q1_direction_is_of_input_subgraph_not_full_graph(self):
        Model,_=self.api();model=Model(0)
        msg=self.message(5,((0,1),(0,2),(1,2),(2,3)),True)
        q=model.q1(msg,msg.ndata['feat'])
        self.assertTrue(torch.equal(q,torch.tensor([1.5,.5,1.,0.,0.])))
        small=self.message(3,((0,1),(0,2)),True)
        self.assertTrue(torch.equal(model.q1(small,small.ndata['feat']),torch.tensor([2.,0.,0.])))
        with self.assertRaises(ValueError):model.logits(msg,torch.zeros(5,32))
        with self.assertRaises(ValueError):model.logits(self.message(2,((0,0),),True),torch.ones(2,32))

    def test_public_reconstruction_after_phi_change_and_projection_commutes(self):
        Model,Layout=self.api();m=Model(2)
        from experiments.node_dp_slots.mechanism import new_model
        from experiments.node_pdp_v2_projected.policy import Layout as OldLayout,project
        with torch.no_grad():m.a.add_(.125);m.b1.sub_(.0625);m.layer3.fc.weight.fill_(-.25)
        reference=new_model(2);reference.load_state_dict(m.reconstruct());oldlayout=OldLayout.from_model(reference)
        message=self.message();x=message.ndata['feat']
        torch.testing.assert_close(m.logits(message,x),reference.logits(message,x),rtol=RTOL,atol=ATOL)
        before=m.vector().clone();Layout.from_model(m).project(m);project(reference,oldlayout)
        for name,v in m.reconstruct().items():self.assertTrue(torch.equal(v,reference.state_dict()[name]),name)
        after=m.vector();self.assertTrue(torch.equal(before[:1185],after[:1185]));self.assertTrue(torch.equal(before[1217:],after[1217:]))

    def test_attention_zero_and_repeated_columns_gradient_chain(self):
        Model,_=self.api()
        from experiments.node_dp_slots.mechanism import new_model
        msg=self.message();x=msg.ndata['feat']
        for seed in range(5):
            old=new_model(seed);new=Model(seed)
            old.logits(msg,x).square().sum().backward();new.logits(msg,x).square().sum().backward()
            g=old.grat.grat1.fc.weight.grad
            torch.testing.assert_close(g,new.a.grad[:,None].expand_as(g),rtol=RTOL,atol=ATOL)
            for p in old.grat.grat1.attn_fc.parameters():self.assertLess(float(p.grad.abs().max()),1e-5)
            self.assertFalse(new.w10.requires_grad);self.assertFalse(new.attn_w0.requires_grad)
            torch.testing.assert_close(old.grat.grat1.fc.bias.grad,new.b1.grad,rtol=RTOL,atol=ATOL)

    def test_relu_zero_derivative_convention(self):
        Model,_=self.api();m=Model(0)
        from experiments.node_dp_slots.mechanism import new_model
        old=new_model(0)
        with torch.no_grad():
            old.grat.grat1.fc.weight.zero_();old.grat.grat1.fc.bias.zero_()
            old.grat.grat1.attn_fc.weight.zero_();old.grat.grat1.attn_fc.bias.zero_();m.a.zero_();m.b1.zero_()
        msg=self.message();x=msg.ndata['feat']
        torch.relu(old.grat.grat1(msg,x)).sum().backward();m.first_hidden(msg,x).sum().backward()
        self.assertEqual(float(old.grat.grat1.fc.weight.grad.abs().max()),0.)
        self.assertEqual(float(m.a.grad.abs().max()),0.);self.assertEqual(float(m.b1.grad.abs().max()),0.)
        self.assertEqual(float(old.grat.grat1.attn_fc.weight.grad.abs().max()),0.)

    def test_new_and_old_clipping_metrics_differ_as_predicted(self):
        self.api()
        from experiments.node_dp_slots.aggregation import clip
        ga=torch.ones(32,dtype=torch.float64);rest=torch.ones(1251,dtype=torch.float64)*.01
        old=torch.cat([ga[:,None].expand(32,32).reshape(-1),rest]);new=torch.cat([ga,rest])
        self.assertAlmostEqual(float(old.square().sum()),32*float(ga.square().sum())+float(rest.square().sum()),places=10)
        self.assertAlmostEqual(float(new.square().sum()),float(ga.square().sum())+float(rest.square().sum()),places=10)
        self.assertGreater(float(clip(new)[-1]),float(clip(old)[-1]))

    def test_independent_spec_and_layout_reject_old_or_changed_model(self):
        Model,Layout=self.api();r,s,g=self.fixture()
        from experiments.rowsum_pdp.config import validate
        from experiments.node_dp_slots.config import Spec as OldSpec
        from experiments.node_pdp.accounting import validate_public
        with self.assertRaises(TypeError):validate(r,OldSpec(s.universe))
        with self.assertRaises(TypeError):validate_public(r,s)
        m=Model(0);m.a=torch.nn.Parameter(torch.ones(31))
        with self.assertRaises(ValueError):Layout.from_model(m)

    def test_k2_slot_locality_and_quota(self):
        self.api();r,s,g=self.fixture()
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_dp_slots.config import memberships
        a=Adapter(s,g);b=Adapter(s,g.delete(3))
        for v in s.universe:self.assertEqual(len(set(memberships(v))),2)
        for j in set(range(4))-set(memberships(3)):self.assertEqual(a.fingerprints()[j],b.fingerprints()[j])
        self.assertEqual([len(items) for items in a.batch()],[4]*4)

    def test_double_clipping_adversarial_twelve_bound(self):
        self.api()
        from experiments.rowsum_pdp.training import double_clip
        from experiments.node_dp_slots.aggregation import clip
        positions=torch.stack([torch.stack([clip(torch.full((1283,),1e8)) for _ in range(4)]) for _ in range(4)])
        signal,slots=double_clip(positions)
        self.assertTrue(all(float(x.norm())<=3 for x in slots))
        opposite=slots.clone();opposite[:2]=-opposite[:2]
        self.assertLessEqual(float((signal-opposite.sum(0)).norm()),12.)
        self.assertGreater(float((signal-opposite.sum(0)).norm()),11.99)
        with self.assertRaises(ValueError):double_clip(torch.zeros(4,4,2340))

    def test_test_gaussian_all1283_scale260(self):
        self.api()
        from experiments.rowsum_pdp.noise import TestGaussian
        calls=[]
        with patch.object(random.Random,'normalvariate',side_effect=lambda mean,std:(calls.append((mean,std)) or 1.)):
            source=TestGaussian(4);vector=source.draw()
        self.assertEqual(calls,[(0.,260.)]*1283);self.assertEqual(vector.shape,(1283,));self.assertEqual(source.calls,1)

    def test_full100_steps_matches_manual_independent_loop(self):
        Model,Layout=self.api();r,s,g=self.fixture()
        from experiments.rowsum_pdp.research import Session
        from experiments.rowsum_pdp.training import center,step
        from experiments.rowsum_pdp.noise import TestGaussian
        from experiments.node_pdp.sampling import sample_once,TestBernoulli
        from experiments.node_dp_slots.adapter import Adapter
        induced=sample_once(g,r,TestBernoulli(17)).graph
        m=Model(3);layout=Layout.from_model(m);adapter=Adapter(s,induced);noise=TestGaussian(19)
        for _ in range(100):
            signal,slots=center(m,adapter.batch());step(m,signal,noise,layout)
        session=Session.synthetic(r,s,model_seed=3,sampling_seed=17,noise_seed=19)
        result=session.run(g)
        self.assertEqual(result.phi,tuple(float(x) for x in m.vector()))
        self.assertEqual(result.diagnostics['updates'],100);self.assertEqual(result.diagnostics['noise_calls'],100)
        self.assertEqual(result.diagnostics['noise_coordinates'],128300);self.assertEqual(result.diagnostics['outer_sampling_calls'],1)
        self.assertTrue(result.diagnostics['all_projected_weights_nonnegative'])
        self.assertGreater(result.diagnostics['nonzero_signal_steps'],0)
        with self.assertRaises(PermissionError):session.run(g)

    def test_empty_graph_still100_and_no_production_controls(self):
        self.api();r,s,_=self.fixture()
        from experiments.rowsum_pdp.research import Session
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import mechanism as prod
        ledger,guard=base._PROCESS_LEDGER,prod._PROCESS_GUARD;before=(ledger.models,guard._used)
        result=Session.synthetic(r,s,model_seed=0,sampling_seed=1,noise_seed=2).run(Graph((),(),True))
        self.assertEqual(result.diagnostics['updates'],100);self.assertEqual(result.diagnostics['nonzero_signal_steps'],0)
        self.assertEqual((ledger.models,guard._used),before);self.assertIs(ledger,base._PROCESS_LEDGER);self.assertIs(guard,prod._PROCESS_GUARD)

    def test_reject_test_sources_in_normal_interface_and_c3_disabled(self):
        self.api();r,s,g=self.fixture()
        from experiments.rowsum_pdp.research import Session
        from experiments.rowsum_pdp.noise import TestGaussian
        session=Session(r,s,model_seed=0);session._noise=TestGaussian(0)
        with self.assertRaisesRegex(PermissionError,'test'):session.run(None)
        session=Session(r,s,model_seed=0)
        with self.assertRaisesRegex(PermissionError,'C2'):session.run(None)

    def test_synthetic_bound_and_ones_domain_rejection(self):
        self.api();r,s,g=self.fixture()
        from experiments.rowsum_pdp.research import Session
        from experiments.rowsum_pdp.config import Spec
        from experiments.node_pdp.registry import Registry
        ids=tuple(range(300));big=Registry(ids,tuple((v,2) for v in ids),source='public-artificial-oversized')
        with self.assertRaises(ValueError):Session.synthetic(big,Spec(ids),model_seed=0,sampling_seed=0,noise_seed=0)

    def test_all_release_paths_reject_candidate(self):
        self.api();r,s,_=self.fixture()
        from experiments.rowsum_pdp.research import Session
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots import release as slots
        from experiments.node_pdp import release as v1
        from experiments.node_pdp_v2_projected import release as v2
        from experiments.rowsum_pdp import release as candidate
        result=Session.synthetic(r,s,model_seed=0,sampling_seed=0,noise_seed=0).run(Graph((),(),True))
        for mod in (slots,v1,v2,candidate):
            with self.assertRaises(PermissionError):mod.export(result,permit_idealized=True)

    def test_independent_rdp_and_amplification_decimal(self):
        self.api();r,s,_=self.fixture()
        from experiments.rowsum_pdp.accounting import base_account,node_accounts
        from decimal import Decimal,localcontext
        with localcontext() as ctx:
            ctx.prec=50;values=[]
            for i in range(11,1000):
                a=Decimal(i)/10;rho=100*a*144/(2*Decimal(260)**2)
                values.append((rho-(Decimal('1e-5').ln()+a.ln())/(a-1)+((a-1)/a).ln(),a))
            eps,order=min(values)
        with patch('experiments.node_dp_slots.accounting.epsilon',side_effect=AssertionError('independent')):
            actual=base_account(s);rows=node_accounts(r,s)
        self.assertAlmostEqual(actual['epsilon'],float(eps),places=12);self.assertEqual(actual['rdp_order'],float(order))
        self.assertTrue(all(x['target_satisfied'] for x in rows))
        from experiments.node_pdp.registry import Registry
        mixed=Registry(s.universe,tuple((v,(1,2,4)[v%3]) for v in s.universe),source='public-artificial-mixed')
        rows=node_accounts(mixed,s);strict=next(x for x in rows if x['epsilon_target']==1)
        self.assertAlmostEqual(strict['epsilon_prime'],.986940353796761,places=12)
        self.assertAlmostEqual(strict['delta_prime'],2.6894142136999514e-6,places=17)


if __name__=='__main__':unittest.main()

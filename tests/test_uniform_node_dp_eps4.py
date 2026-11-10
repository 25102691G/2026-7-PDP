"""U2 artificial mechanisms only; no prepared graph/model/real evaluation access."""
import dataclasses
import math
import random
import unittest
from unittest.mock import patch
import torch


class UniformEps4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self,n=12,directed=True):
        from experiments.uniform_node_dp_eps4.config import Spec
        from experiments.node_pdp.registry import Registry
        from experiments.node_dp_slots.adapter import Graph
        ids=tuple(range(n))
        return Spec(ids,directed),Registry(ids,tuple((v,(1,2,4)[v%3]) for v in ids),source='public-artificial-U2'),Graph(ids,tuple((v,v+1) for v in range(n-1)),directed)

    def flat(self,model):return torch.cat([p.detach().flatten() for p in model.parameters()])

    def test_frozen_configuration_rejects_each_wrong_mechanism_parameter(self):
        from experiments.uniform_node_dp_eps4.config import Spec,VERSION,validate
        self.assertEqual(VERSION,'uniform_node_dp_eps4_v1')
        wrong=dict(s_sum=138.,T=101,target_delta=1e-4,sensitivity_sum=13.,dimension=33,p_v=.5,
                   projection='initial projection',initial_projection=True,B=15,C=2.,K=1,lr=.01,momentum=.1,target_epsilon=1.)
        for key,value in wrong.items():
            with self.subTest(key=key),self.assertRaises(ValueError):Spec((0,1),**{key:value})
        from experiments.uniform_node_dp.config import Spec as Old
        with self.assertRaises(TypeError):validate(Old((0,1)))
        s,_,_=self.fixture()
        with self.assertRaises(dataclasses.FrozenInstanceError):s.s_sum=486.
        self.assertEqual(s.public()['target_epsilon'],4.)
        self.assertEqual(s.public()['target_delta'],1e-5)
        self.assertEqual(s.public()['s_mean'],8.6875)

    def test_all2340_initial_parameters_match_original_uniform_for_seeds0to4(self):
        from experiments.uniform_node_dp_eps4.research import Session
        from experiments.uniform_node_dp.research import Session as Old
        from experiments.uniform_node_dp.config import Spec as OldSpec
        s,r,g=self.fixture()
        for seed in range(5):
            a=Session.synthetic(s,r,noise_seed=0,model_seed=seed);b=Old.synthetic(OldSpec(s.universe),noise_seed=1,model_seed=seed)
            self.assertEqual(sum(p.numel() for p in a.model.parameters()),2340)
            self.assertEqual(sum(p.numel() for p in a.model.parameters() if p.requires_grad),2340)
            self.assertEqual(len(list(a.model.parameters())),12)
            self.assertEqual([type(x).__name__ for x in [a.model.grat.grat1,a.model.grat.grat2,a.model.grat.grat3]],['LegacyGRATLayer']*3)
            for (n,p),(m,q) in zip(a.model.named_parameters(),b.model.named_parameters()):
                self.assertEqual((n,tuple(p.shape)),(m,tuple(q.shape)));self.assertTrue(torch.equal(p,q))
            self.assertTrue(bool((a.layout.target(a.model)<0).any()))

    def test_forward_loss_raw_gradient_and_both_clips_equal_original(self):
        from experiments.uniform_node_dp_eps4.research import Session
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.adapter import Item
        from experiments.node_dp_slots.aggregation import clip,read_gradient,double_clip
        s,r,g=self.fixture();a=Session.synthetic(s,r,noise_seed=0).model;b=base.new_model(0)
        _,Loss,to_dgl,to_adj,ig=base.runtime()
        for edges in [((0,1),(1,2),(2,0)),(),((0,2),(1,2))]:
            grads=[];outputs=[];losses=[]
            for model in (a,b):
                model.zero_grad(set_to_none=True);graph=ig.Graph(n=4,edges=edges,directed=True)
                message=to_dgl(graph,'ones',32,True,False);out=model(message,message.ndata['feat']).squeeze(1)
                loss=Loss(.5)(out,to_adj(graph),50);loss.backward()
                grads.append(read_gradient(list(model.parameters())));outputs.append(out.detach());losses.append(float(loss))
            self.assertTrue(torch.equal(outputs[0],outputs[1]));self.assertEqual(losses[0],losses[1])
            self.assertTrue(torch.equal(grads[0],grads[1]));self.assertGreater(float(grads[0].norm()),0.)
            positions=torch.zeros(4,4,2340,dtype=torch.float64);positions[0,0]=clip(grads[0])
            expected,slots=double_clip(positions)
            actual,got=base.center(a,[[Item((0,1,2,3),edges,True),None,None,None]]+[[None]*4 for _ in range(3)])
            self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(got,slots))

    def test_four_slots_node_add_remove_locality_and_tight12_bound(self):
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_dp_slots.config import memberships
        from experiments.node_dp_slots.aggregation import clip,double_clip
        s,r,g=self.fixture();a=Adapter(s,g);b=Adapter(s,g.delete(0))
        self.assertIn(1,g.delete(0).nodes)
        self.assertEqual(len(a.caches),4);self.assertEqual([len(x) for x in a.batch()],[4]*4)
        for v in s.universe:self.assertEqual(len(set(memberships(v))),2)
        for j in set(range(4))-set(memberships(0)):
            self.assertEqual(Adapter(s,g).fingerprints()[j],b.fingerprints()[j])
        raw=torch.zeros(4,4,2340,dtype=torch.float64);raw[:,:,0]=1e8
        x=torch.stack([torch.stack([clip(v) for v in row]) for row in raw]);signal,slots=double_clip(x)
        other=slots.clone();other[:2]=-other[:2]
        self.assertLessEqual(float(x.norm(dim=2).max()),3.)
        self.assertLessEqual(float(slots.norm(dim=1).max()),3.)
        self.assertGreater(float((signal-other.sum(0)).norm()),11.99)
        self.assertLessEqual(float((signal-other.sum(0)).norm()),12.)

    def test_noise_is_2340_independent_coordinate_calls_at139_not_last_layer_only(self):
        from experiments.uniform_node_dp_eps4.noise import SecretGaussian,TestGaussian
        calls=[]
        with patch.object(random.SystemRandom,'normalvariate',side_effect=lambda mu,sd:calls.append((mu,sd)) or 0.):
            source=SecretGaussian();z=source.draw()
        self.assertEqual(z.shape,(2340,));self.assertEqual(z.dtype,torch.float64)
        self.assertEqual(calls,[(0.,139.)]*2340);self.assertEqual(source.calls,1)
        a=TestGaussian(8).draw();rng=random.Random(8)
        self.assertEqual(a.tolist(),[rng.normalvariate(0.,139.) for _ in range(2340)])
        self.assertTrue(bool((a[:2242]!=0).all()));self.assertNotEqual(float(a[-1]),0.)
        self.assertNotEqual(TestGaussian(8).draw().tolist(),TestGaussian(9).draw().tolist())

    def test_update_scales_signal_and_noise_once_by16_then_projects_only32(self):
        from experiments.uniform_node_dp_eps4.research import Session
        from experiments.node_pdp_v2_projected import policy
        from experiments.node_dp_slots import mechanism as base
        s,r,g=self.fixture();session=Session.synthetic(s,r,noise_seed=0)
        a=base.new_model(4);b=base.new_model(4)
        with torch.no_grad():a.grat.grat3.fc.bias.fill_(-10);b.grat.grat3.fc.bias.fill_(-10)
        old=self.flat(a).double();signal=torch.zeros(2340,dtype=torch.float64);signal[0]=12.
        z=torch.linspace(-139.,139.,2340,dtype=torch.float64)
        class PublicFixed:
            calls=0
            def draw(self):self.calls+=1;return z.clone()
        source=PublicFixed();policy.step(a,signal,source,session.layout)
        expected=(old-.005*(signal+z)/16).float();lo,hi=session.layout.start,session.layout.stop
        expected[lo:hi].clamp_min_(0)
        self.assertTrue(torch.equal(self.flat(a),expected));self.assertEqual(source.calls,1)
        base.update(b.parameters(),signal,PublicFixed());unprojected=self.flat(b)
        self.assertTrue(torch.equal(self.flat(a)[:lo],unprojected[:lo]))
        self.assertTrue(torch.equal(self.flat(a)[hi:],unprojected[hi:]));self.assertLess(float(a.grat.grat3.fc.bias),0.)
        self.assertEqual(139/16,8.6875);self.assertEqual(12/16,.75)

    def test_dense_artificial_100step_loop_noise_and_every_projection(self):
        from experiments.uniform_node_dp_eps4.research import Session
        from experiments.uniform_node_dp_eps4.noise import TestGaussian
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.adapter import Graph,Adapter
        from experiments.node_pdp_v2_projected import policy
        s,r,g=self.fixture(96);g=Graph(g.nodes,tuple((u,v) for u in g.nodes for v in g.nodes if u!=v),True)
        session=Session.synthetic(s,r,noise_seed=17);batch=Adapter(s,g).batch()
        self.assertTrue(any(i is not None for row in batch for i in row))
        signal,_=base.center(session.model,batch);self.assertGreater(float(signal.norm()),0.)
        shapes=[];scales=[];projections=[];draw=TestGaussian.draw;project=policy.project;normal=random.Random.normalvariate
        def observe_draw(source):
            z=draw(source);shapes.append(tuple(z.shape));return z
        def observe_normal(rng,mu,sd):scales.append((mu,sd));return normal(rng,mu,sd)
        def observe_projection(model,layout):
            bias=model.grat.grat3.fc.bias.detach().clone();info=project(model,layout)
            self.assertTrue(torch.equal(model.grat.grat3.fc.bias,bias));self.assertTrue(info['nonnegative'])
            projections.append(1);return info
        with patch.object(TestGaussian,'draw',observe_draw),patch.object(random.Random,'normalvariate',observe_normal),patch.object(policy,'project',observe_projection),patch('experiments.node_pdp.sampling.sample_once',side_effect=AssertionError('no outer Bernoulli')):
            result=session.run(g)
        self.assertEqual(shapes,[(2340,)]*100);self.assertEqual(scales,[(0.,139.)]*234000);self.assertEqual(len(projections),100)
        self.assertEqual(result.diagnostics['noise_coordinates'],234000)
        self.assertEqual(result.diagnostics['outer_sampling_calls'],0)
        self.assertEqual(result.configuration['algorithm'],'uniform_node_dp_eps4_v1')
        self.assertEqual(result.configuration['account']['target_epsilon'],4.)
        self.assertFalse(result.configuration['machine_dp_certified'])
        with self.assertRaises(PermissionError):session.run(g)

    def test_empty_isolated_directed_and_undirected_inputs_keep100updates(self):
        from experiments.uniform_node_dp_eps4.research import Session
        from experiments.node_dp_slots.adapter import Graph
        for directed in [True,False]:
            s,r,g=self.fixture(directed=directed)
            for nodes,edges in [((),()),(g.nodes,()),(g.nodes,((0,1),(1,2)))]:
                result=Session.synthetic(s,r,noise_seed=5).run(Graph(nodes,edges,directed))
                self.assertEqual(result.diagnostics['updates'],100)
                self.assertEqual(result.diagnostics['noise_calls'],100)
                self.assertEqual(result.diagnostics['projection_calls'],100)
                self.assertTrue(all(math.isfinite(v) for v in result.parameters))

    def test_accounting_70digit_independent_decimal_and_exact_certificate(self):
        from decimal import Decimal,localcontext
        from fractions import Fraction as F
        from experiments.uniform_node_dp_eps4.accounting import account,exact_certificate
        s,r,g=self.fixture()
        with localcontext() as c:
            c.prec=70;values=[]
            for k in range(11,1000):
                a=Decimal(k)/10;rho=100*a*144/(2*Decimal(139)**2)
                values.append((rho-(Decimal('1e-5').ln()+a.ln())/(a-1)+((a-1)/a).ln(),a))
            expected,alpha=min(values)
        with patch('experiments.node_pdp.accounting.amplify',side_effect=AssertionError('no PDP amplification')):
            actual=account(s)
        self.assertAlmostEqual(actual['epsilon_numeric'],float(expected),places=13)
        self.assertEqual(actual['rdp_order'],float(alpha));self.assertEqual(actual['delta'],1e-5)
        self.assertLessEqual(actual['epsilon_upper_bound'],4.)
        self.assertGreater(actual['epsilon_upper_bound'],1.)
        self.assertEqual(actual['target_epsilon'],4.)
        cert=exact_certificate(s);lo=F(cert['lower_fraction']);hi=F(cert['upper_fraction'])
        self.assertLess(hi,F('3.997'));self.assertLess(hi-lo,F(1,10**20));self.assertLessEqual(lo,hi)

    def test_accountant_refuses_insufficient_noise_and_invalid_numeric_domain(self):
        from experiments.uniform_node_dp_eps4.accounting import gaussian_bound,require_target
        self.assertGreater(gaussian_bound(sigma_sum=138.)['epsilon_numeric'],4.)
        with self.assertRaises(ValueError):require_target(gaussian_bound(sigma_sum=138.),4.)
        for value in [0.,-1.,math.nan,math.inf]:
            with self.subTest(value=value),self.assertRaises(ValueError):gaussian_bound(sigma_sum=value)
        for kwargs in [dict(steps=0),dict(delta=0.),dict(delta=1.),dict(sensitivity=-1.),dict(dimension=33),dict(p_v=.5)]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):gaussian_bound(**kwargs)

    def test_extreme_numeric_scales_never_turn_nonfinite_rdp_into_zero_privacy(self):
        from experiments.uniform_node_dp_eps4.accounting import gaussian_bound,require_target
        bound=gaussian_bound(sigma_sum=1e154,sensitivity=1e154)
        self.assertGreater(bound['epsilon_numeric'],4.)
        with self.assertRaises(ValueError):require_target(bound,4.)
        for scale in [1e-320,1e308]:
            with self.subTest(scale=scale),self.assertRaises(ValueError):gaussian_bound(sigma_sum=scale)

    def test_target_guard_rejects_nan_inf_and_negative_epsilon(self):
        from experiments.uniform_node_dp_eps4.accounting import require_target
        for epsilon in [math.nan,math.inf,-1.]:
            with self.subTest(epsilon=epsilon),self.assertRaises(ValueError):require_target({'epsilon_numeric':epsilon},4.)

    def test_all_registry_identities_have_uniform4_bound_not_original_strict_caps(self):
        from experiments.uniform_node_dp_eps4.accounting import node_accounts
        s,r,g=self.fixture();rows=node_accounts(r,s)
        self.assertEqual(len(rows),len(r.universe))
        self.assertEqual({x['p_v'] for x in rows},{1.})
        self.assertEqual(len({(x['epsilon_upper_bound'],x['delta']) for x in rows}),1)
        self.assertTrue(all(x['target_epsilon']==4. for x in rows))
        self.assertEqual({x['original_policy_target'] for x in rows},{1,2,4})
        self.assertEqual({x['original_policy_certified'] for x in rows if x['original_policy_target'] in (1,2)},{False})

    def test_tampered_public_state_registry_and_wrong_domain_fail_before_graph(self):
        from experiments.uniform_node_dp_eps4.research import Session
        s,r,g=self.fixture()
        for alteration in ['freeze','initial','layout','spec']:
            session=Session.synthetic(s,r,noise_seed=7)
            if alteration=='freeze':next(session.model.parameters()).requires_grad_(False)
            if alteration=='initial':
                with torch.no_grad():next(session.model.parameters()).add_(1)
            if alteration=='layout':session._layout=dataclasses.replace(session.layout,total=33)
            if alteration=='spec':object.__setattr__(session._spec,'s_sum',138.)
            with patch('experiments.node_dp_slots.mechanism.Adapter',side_effect=AssertionError('premature graph access')):
                with self.subTest(alteration=alteration),self.assertRaises(ValueError):session.run(g)
            s,r,g=self.fixture()
        from experiments.node_pdp.registry import Registry
        with self.assertRaises(ValueError):Session.synthetic(s,Registry((0,),((0,1),),source='artificial'),noise_seed=7)
        from experiments.node_dp_slots.adapter import Graph
        with self.assertRaises(ValueError):Session.synthetic(s,r,noise_seed=7).run(Graph((999,),(),True))

    def test_real_run_disabled_before_graph_or_secret_noise_and_cli_fails_closed(self):
        from experiments.uniform_node_dp_eps4.research import Session
        from experiments.uniform_node_dp_eps4.__main__ import main
        s,r,g=self.fixture()
        with patch('experiments.node_dp_slots.mechanism.Adapter',side_effect=AssertionError('private graph access')),patch('experiments.uniform_node_dp_eps4.noise.SecretGaussian.draw',side_effect=AssertionError('secret randomness')):
            with self.assertRaisesRegex(PermissionError,'REAL_TRAINING_DISABLED'):Session(s,r).run(g)
            with self.assertRaisesRegex(PermissionError,'REAL_TRAINING_DISABLED'):main([])
            with self.assertRaisesRegex(PermissionError,'REAL_TRAINING_DISABLED'):main(['--dataset','email'])

    def test_release_rejects_new_result_and_historical_privim_is_not_recognized(self):
        from types import SimpleNamespace
        from experiments.uniform_node_dp_eps4.research import Session,validate_result
        from experiments.uniform_node_dp_eps4 import release
        from experiments.node_dp_slots import release as slots
        from experiments.node_pdp import release as pdp
        from experiments.node_pdp_v2_projected import release as v2
        from experiments.matched_pdp_frozen_backbone import release as matched
        s,r,g=self.fixture();result=Session.synthetic(s,r,noise_seed=2).run(g);validate_result(result)
        for api in [release,slots,pdp,v2,matched]:
            with self.assertRaises(PermissionError):api.export(result,permit_idealized=True)
        historical=SimpleNamespace(parameters=result.parameters,configuration={'algorithm':'PrivIM*','target_epsilon':4})
        with self.assertRaises(TypeError):validate_result(historical)
        fake=dataclasses.replace(result,configuration={**result.configuration,'algorithm':'PrivIM*'})
        with self.assertRaises(ValueError):validate_result(fake)

    def test_synthetic_result_does_not_record_random_material_or_touch_ledgers(self):
        import json
        from experiments.uniform_node_dp_eps4.research import Session
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import mechanism as pdp
        s,r,g=self.fixture();before=(base._PROCESS_LEDGER.models,pdp._PROCESS_GUARD._used)
        result=Session.synthetic(s,r,noise_seed=17).run(g)
        self.assertEqual(before,(base._PROCESS_LEDGER.models,pdp._PROCESS_GUARD._used))
        text=json.dumps({'configuration':result.configuration,'diagnostics':result.diagnostics})
        for key in ['noise_seed','noise_vector','random_bytes','retained_mask']:
            self.assertNotIn(key,text)


if __name__=='__main__':unittest.main()

"""Artificial parameters/graphs only; no prepared input or historical model access."""
import importlib.util
import json
import math
import unittest
from unittest.mock import patch

import torch


class ProjectedV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.node_pdp_v2_projected'))
        from experiments.node_pdp_v2_projected import policy,accounting,mechanism,release,diagnostics
        return policy,accounting,mechanism,release,diagnostics

    def model(self):
        from experiments.node_dp_slots.mechanism import new_model
        return new_model()

    def flat(self,model):return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def fixture(self,directed=True):
        from experiments.node_dp_slots.config import Spec
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_pdp.registry import Registry
        ids=tuple(range(128));spec=Spec(ids,directed)
        reg=Registry(ids,tuple((v,2) for v in ids),source='public-artificial-v2')
        edges=[(i,(i+k)%96) for i in range(96) for k in range(1,13)]
        if not directed:edges=sorted(set(tuple(sorted(e)) for e in edges))
        return reg,spec,Graph(tuple(range(96)),tuple(edges),directed)

    def privacy_context(self):
        from tests.test_node_pdp import NodePDPTests
        return NodePDPTests().privacy_context()

    def constant_model(self,hidden_bias=1.,last_weight=1.):
        model=self.model()
        with torch.no_grad():
            for layer in (model.grat.grat1,model.grat.grat2,model.grat.grat3):
                layer.fc.weight.zero_();layer.fc.bias.fill_(hidden_bias)
                layer.attn_fc.weight.zero_();layer.attn_fc.bias.zero_()
            model.grat.grat3.fc.weight.fill_(last_weight);model.grat.grat3.fc.bias.fill_(.25)
        model.eval();return model

    def probe(self,model,graph):
        from experiments.graph_util import get_rev_dgl
        message=get_rev_dgl(graph,'ones',32,graph.is_directed(),False)
        capture={}
        def record(module,args):capture['aggregate']=args[0].detach().clone()
        handle=model.grat.grat3.fc.register_forward_pre_hook(record)
        try:
            with torch.no_grad():scores=model.logits(message,message.ndata['feat']).reshape(-1)
        finally:handle.remove()
        return message,capture['aggregate'],scores

    def test_layout_is_discovered_from_named_parameters(self):
        p,*_=self.api();model=self.model();layout=p.Layout.from_model(model)
        self.assertEqual(layout.target_name,'grat.grat3.fc.weight')
        self.assertEqual(layout.shape,(1,32));self.assertEqual(layout.total,2340)
        # Expected observations of this architecture, not offsets used by the policy.
        self.assertEqual((layout.start,layout.stop),(2242,2274))
        self.assertEqual(layout.stop-layout.start,32)
        self.assertTrue(bool((model.grat.grat3.fc.weight<0).any()))
        self.assertEqual(sum(row['coordinates'] for row in layout.public()['parameters']),2340)

    def test_projection_only_changes_target_not_bias_or_other_2308(self):
        p,*_=self.api();model=self.model();layout=p.Layout.from_model(model)
        with torch.no_grad():model.grat.grat3.fc.weight.copy_(torch.linspace(-2,2,32).reshape(1,32))
        before=self.flat(model).clone();stats=p.project(model,layout);after=self.flat(model)
        mask=torch.ones(2340,dtype=torch.bool);mask[layout.start:layout.stop]=False
        self.assertTrue(torch.equal(before[mask],after[mask]))
        self.assertTrue(torch.equal(after[layout.start:layout.stop],before[layout.start:layout.stop].clamp_min(0)))
        self.assertEqual(stats['clamped_coordinates'],16);self.assertGreater(stats['displacement_l2'],0)
        positive=after.clone();p.project(model,layout);self.assertTrue(torch.equal(positive,self.flat(model)))

    def test_noise_update_repair_then_projection_matches_independent_formula(self):
        p,*_=self.api();model=self.model();layout=p.Layout.from_model(model)
        before=self.flat(model).double();signal=torch.zeros(2340,dtype=torch.float64);signal[0]=12
        z=torch.linspace(-260,260,2340,dtype=torch.float64)
        class Fixed:
            calls=0
            def draw(self):self.calls+=1;return z.clone()
        source=Fixed();processed=(before-.005*(signal+z)/16).float()
        expected=processed.clone();expected[layout.start:layout.stop].clamp_min_(0)
        p.step(model,signal,source,layout)
        self.assertEqual(source.calls,1);self.assertTrue(torch.equal(self.flat(model),expected))
        self.assertNotEqual(float(processed[layout.stop]),float(before[layout.stop]))
        self.assertEqual(float(self.flat(model)[layout.stop]),float(processed[layout.stop]))

    def test_constrained_coordinates_still_receive_noise(self):
        p,*_=self.api();model=self.model();layout=p.Layout.from_model(model)
        with torch.no_grad():model.grat.grat3.fc.weight.zero_()
        class Fixed:
            def draw(self):
                z=torch.zeros(2340,dtype=torch.float64);z[layout.start:layout.stop]=-260.;return z
        p.step(model,torch.zeros(2340),Fixed(),layout)
        self.assertTrue(torch.equal(model.grat.grat3.fc.weight,torch.full((1,32),.08125)))

    def test_nonfinite_processing_precedes_projection(self):
        p,*_=self.api();model=self.model();layout=p.Layout.from_model(model)
        class Bad:
            def draw(self):
                z=torch.zeros(2340);z[layout.start]=-float('inf');z[0]=float('nan');z[layout.stop]=float('inf');return z
        p.step(model,torch.zeros(2340),Bad(),layout);flat=self.flat(model)
        self.assertTrue(bool(torch.isfinite(flat).all()))
        self.assertEqual(float(flat[layout.start]),0.);self.assertEqual(float(flat[0]),0.);self.assertEqual(float(flat[layout.stop]),0.)

    def test_directed_nonnegative_aggregate_margin_and_sink_bias(self):
        p,*_=self.api();import igraph
        graph=igraph.Graph(n=3,edges=[(0,1),(1,2)],directed=True)
        model=self.constant_model(last_weight=-1.);layout=p.Layout.from_model(model)
        _,_,before=self.probe(model,graph);self.assertTrue(bool((before<.25).any()))
        p.project(model,layout)
        with torch.no_grad():model.grat.grat3.fc.weight.fill_(1.)
        message,m,scores=self.probe(model,graph)
        self.assertEqual(message.in_degrees().tolist(),graph.outdegree())
        self.assertTrue(bool((m>=0).all()));self.assertEqual(float(scores[2]),.25)
        self.assertTrue(bool((scores[:2]>.25).all()))

    def test_undirected_isolated_node_aggregate_zero(self):
        self.api();import igraph
        graph=igraph.Graph(n=3,edges=[(0,1)],directed=False)
        message,m,scores=self.probe(self.constant_model(),graph)
        self.assertEqual(message.in_degrees().tolist(),graph.outdegree())
        self.assertTrue(bool((m>=0).all()));self.assertTrue(bool((m[2]==0).all()));self.assertEqual(float(scores[2]),.25)

    def test_zero_weight_or_zero_hidden_messages_allow_ties(self):
        self.api();import igraph
        graph=igraph.Graph(n=3,edges=[(0,1),(1,0)],directed=True)
        for bias,weight in ((1.,0.),(-1.,1.)):
            _,m,scores=self.probe(self.constant_model(bias,weight),graph)
            self.assertTrue(bool((scores==.25).all()))
            if bias<0:self.assertTrue(bool((m==0).all()))

    def test_projected_synthetic_top50_can_still_have_spread50(self):
        p,*_=self.api();import igraph
        from baselines.heuristics import get_influence
        graph=igraph.Graph(n=128,edges=[(50,v) for v in range(128) if v!=50]+[(v,50) for v in range(51,128)],directed=True)
        model=self.constant_model(-1.,1.);p.project(model,p.Layout.from_model(model))
        _,m,scores=self.probe(model,graph)
        selected=torch.argsort(scores,descending=True,stable=True)[:50].tolist()
        self.assertTrue(bool((m==0).all()));self.assertEqual(selected,list(range(50)))
        self.assertEqual(get_influence(graph,selected),50)
        self.assertEqual(get_influence(graph,[50]+list(range(49))),128)

    def test_accounting_independently_matches_v1_and_all_targets(self):
        _,a,*_=self.api();reg,spec,_=self.fixture()
        from experiments.node_pdp.registry import Registry
        reg=Registry((0,1,2),((0,1),(1,2),(2,4)),source='public-artificial')
        from experiments.node_dp_slots.config import Spec
        account=a.base_account(Spec(reg.universe));self.assertAlmostEqual(account['epsilon'],1.982092738991924,places=14)
        self.assertEqual(account['rdp_order'],10.3)
        rows=a.node_accounts(reg,Spec(reg.universe))
        self.assertAlmostEqual(rows[0]['epsilon_prime'],.9869403537967609,places=14)
        self.assertAlmostEqual(rows[0]['delta_prime'],2.6894142136999514e-6,places=18)
        self.assertTrue(all(row['target_satisfied'] for row in rows))

    def test_common_history_projected_states_keep_conditional_slot_locality(self):
        p,*_=self.api();reg,spec,graph=self.fixture()
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_dp_slots.config import memberships
        left,right=Adapter(spec,graph),Adapter(spec,graph.delete(47))
        model=self.model();layout=p.Layout.from_model(model)
        class History:
            def draw(self):return torch.linspace(-1.,1.,2340,dtype=torch.float64)
        for _ in range(2):
            p.step(model,torch.zeros(2340),History(),layout)  # same prescribed Y history
            x,bx=base.center(model,left.batch());y,by=base.center(model,right.batch())
            self.assertLessEqual(float((x-y).norm()),12.)
            for j in set(range(4))-set(memberships(47)):self.assertTrue(torch.equal(bx[j],by[j]))
            self.assertTrue(all(float(b.norm())<=3 for b in bx))

    def test_full_synthetic_training_100_updates_no_initial_projection(self):
        p,_,m,_,_=self.api();reg,spec,graph=self.fixture()
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import sampling
        initial=self.flat(self.model());engine=m.Engine.testing(reg,spec,sampling_seed=0,noise_seed=0)
        self.assertTrue(torch.equal(initial,self.flat(engine._model)))
        original=base.update
        with patch.object(sampling,'sample_once',wraps=sampling.sample_once) as sample,patch.object(base,'update',wraps=original) as update:
            result=engine.run(graph)
        self.assertEqual(sample.call_count,1)
        self.assertEqual(update.call_count,100)
        d=result.testing_diagnostics
        self.assertEqual(d['updates'],100);self.assertEqual(d['noise_calls'],100)
        self.assertEqual(d['projection_calls'],100);self.assertEqual(d['noise_coordinates'],234000)
        self.assertTrue(d['all_projected_weights_nonnegative']);self.assertTrue(d['all_parameters_finite'])
        self.assertIn('joint slot contribution exactly zero after slot clipping',d['zero_gradient_slot_definition'])
        self.assertTrue(all(x['positions']==16 for x in d['steps']))
        self.assertTrue(any(x['effective_positions'] for x in d['steps']))
        self.assertEqual(result._spec.universe,spec.universe)
        self.assertEqual(result._registry_fingerprint,reg.fingerprint)
        with self.assertRaises(PermissionError):engine.run(graph)

    def test_empty_graph_padding_still_has_100_noisy_projected_updates(self):
        _,_,m,_,_=self.api();reg,spec,graph=self.fixture()
        from experiments.node_dp_slots.adapter import Graph
        d=m.Engine.testing(reg,spec,sampling_seed=0,noise_seed=1).run(Graph((),(),True)).testing_diagnostics
        self.assertEqual(d['updates'],100)
        self.assertTrue(all(x['effective_positions']==0 and x['empty_candidate_slots']==4 and x['zero_gradient_slots']==4 for x in d['steps']))

    def test_short_local_containers_pad_without_cross_slot_topup(self):
        _,_,m,_,_=self.api();reg,spec,graph=self.fixture()
        original=m.base.Adapter
        def short(*args):
            obj=original(*args)
            for cache in obj.caches:cache.subgraphs=cache.subgraphs[:2]
            return obj
        with patch.object(m.base,'Adapter',side_effect=short):
            d=m.Engine.testing(reg,spec,sampling_seed=0,noise_seed=0).run(graph).testing_diagnostics
        self.assertTrue(all(s['positions']==16 and s['effective_positions']==8 for s in d['steps']))
        self.assertEqual(d['updates'],100)

    def test_changed_layout_refused_before_noise_or_update(self):
        p,*_=self.api();from dataclasses import replace
        model=self.model();layout=replace(p.Layout.from_model(model),start=0);before=self.flat(model).clone()
        class Never:
            def draw(self):raise AssertionError('noise accessed before public layout validation')
        with self.assertRaises(ValueError):p.step(model,torch.zeros(2340),Never(),layout)
        self.assertTrue(torch.equal(before,self.flat(model)))

    def test_failure_does_not_refund_cross_version_authorization(self):
        _,_,m,_,_=self.api();reg,spec,_=self.fixture()
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_pdp import mechanism as v1
        with self.privacy_context():
            with patch.object(m.base,'center',side_effect=RuntimeError('synthetic private detail')):
                with self.assertRaises(v1.ExecutionGap) as caught:m.Engine.privacy(reg,spec).run(Graph((),(),True))
            self.assertNotIn('private detail',str(caught.exception))
            with self.assertRaises(PermissionError):v1.Engine.privacy(reg,spec).run(None)

    def test_nonempty_candidates_do_not_imply_nonzero_slot_gradients(self):
        *_,d=self.api()
        info=d.slot_summary([[object()]*4 for _ in range(4)],torch.zeros(4,2340))
        self.assertEqual(info['effective_positions'],16);self.assertEqual(info['empty_candidate_slots'],0)
        self.assertEqual(info['zero_gradient_slots'],4)

    def test_privacy_mode_release_version_and_shared_v1_budget(self):
        _,_,m,r,_=self.api();reg,spec,graph=self.fixture()
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_pdp import mechanism as v1,release as v1release
        with self.privacy_context():
            result=m.Engine.privacy(reg,spec).run(Graph((),(),True))
            self.assertIsNone(result.testing_diagnostics)
            with self.assertRaises(PermissionError):r.export(result)
            with self.assertRaises(PermissionError):v1release.export(result,permit_idealized=True)
            payload=r.export(result,permit_idealized=True)
            self.assertEqual(set(payload),{'parameters','configuration'})
            self.assertEqual(payload['configuration']['protocol'],'node_pdp_v2_projected')
            self.assertEqual(payload['configuration']['projection']['start'],2242)
            self.assertFalse(payload['configuration']['machine_pdp_certified'])
            self.assertNotIn('steps',payload['configuration'])
            with self.assertRaises(PermissionError):r.export(result,permit_idealized=True)
            with self.assertRaises(PermissionError):v1.Engine.privacy(reg,spec).run(None)
            with self.assertRaises(PermissionError):m.Engine.privacy(reg,spec).run(None)

    def test_v1_first_blocks_v2_before_private_access(self):
        _,_,m,_,_=self.api();reg,spec,_=self.fixture()
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_pdp.mechanism import Engine as V1
        with self.privacy_context():
            V1.privacy(reg,spec).run(Graph((),(),True))
            with patch.object(m.base,'Adapter',side_effect=AssertionError('private graph accessed')):
                with self.assertRaises(PermissionError):m.Engine.privacy(reg,spec).run(None)

    def test_testing_sources_and_testing_results_are_not_publishable(self):
        _,_,m,r,_=self.api();reg,spec,graph=self.fixture()
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_pdp.sampling import TestBernoulli
        from experiments.node_dp_slots.noise import TestGaussian
        test=m.Engine.testing(reg,spec,sampling_seed=0,noise_seed=0).run(Graph((),(),True))
        with self.assertRaises(PermissionError):r.export(test,permit_idealized=True)
        for outer in (True,False):
            with self.privacy_context():
                engine=m.Engine.privacy(reg,spec)
                if outer:engine._sampler=TestBernoulli(0)
                else:engine._carrier._source=TestGaussian(0)
                with self.assertRaises(PermissionError):engine.run(None)

    def test_package_has_no_real_dataset_or_checkpoint_loader(self):
        self.api();import ast
        from pathlib import Path
        for path in Path('experiments/node_pdp_v2_projected').glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.Call):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                    self.assertNotIn(name,('load_prepared_graph','load_prepared_train','load_saved_case','load_state_dict','load'))


if __name__=='__main__':unittest.main()

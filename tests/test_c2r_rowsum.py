"""Artificial C2-R only; tolerance frozen by C2, not chosen from results."""
import importlib.util
import json
from pathlib import Path
import unittest
import torch
from experiments.node_dp_slots.mechanism import runtime,new_model

RTOL=ATOL=2e-5
OUT=Path('docs/pdp_research/stage_c2r_rowsum')


def message(n,edges=(),directed=True):
    graph=runtime()[4].Graph(n=n,edges=edges,directed=directed)
    return runtime()[2](graph,'ones',32,directed,False)


class SemanticsTest(unittest.TestCase):
    def test_c2_counterexample_reproduced(self):
        from experiments.rowsum_pdp.model import Model as FailedModel
        model=new_model(0);failed=FailedModel(0);graph=message(4);x=graph.ndata['feat'];captured=[]
        hook=model.grat.grat1.fc.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
        with torch.no_grad():old=model.logits(graph,x);bad=failed.logits(graph,x)
        hook.remove()
        self.assertTrue(torch.equal(captured[0],torch.ones(4,32)))
        difference=float((old-bad).abs().max())
        self.assertAlmostEqual(difference,.004411786794662476,places=12)
        self.assertGreater(difference,ATOL)
        (OUT/'c2_counterexample_reproduced.json').write_text(json.dumps(dict(nodes=4,edges=0,public_model_seed=0,old_logits=old.reshape(-1).tolist(),old_c2_logits=bad.reshape(-1).tolist(),max_absolute_error=difference,original_first_aggregate='ones32',frozen_tolerance=ATOL),indent=2)+'\n')

    def test_revised_entry_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.rowsum_pdp_c2r'),'independent C2-R module required')

import itertools
from unittest.mock import patch
import test_rowsum_pdp as old_tests


class RevisedRowSumTests(old_tests.RowSumTests):
    def api(self):
        from experiments.rowsum_pdp_c2r.model import Model,Layout
        return Model,Layout

    def test_full100_steps_matches_manual_independent_loop(self):
        from experiments.rowsum_pdp_c2r.research import Session
        from experiments.rowsum_pdp import training
        registry,spec,graph=self.fixture();observed=[];public_buffers={}
        real_step=training.step
        def checked_step(model,signal,source,layout):
            before=model.vector().double().clone();buffers={n:p.clone() for n,p in model.named_buffers()}
            real_draw=source.draw;packets=[]
            def traced_draw():
                value=real_draw();packets.append(value.clone());return value
            with patch.object(source,'draw',side_effect=traced_draw):result=real_step(model,signal,source,layout)
            self.assertEqual(len(packets),1);self.assertEqual(packets[0].shape,(1283,))
            expected=(before-.005*(signal.double()+packets[0])/16).float()
            expected=torch.where(torch.isfinite(expected),expected,torch.zeros_like(expected));expected[1185:1217].clamp_min_(0.)
            self.assertTrue(torch.equal(model.vector(),expected))
            self.assertTrue(result['nonnegative']);self.assertLessEqual(float(signal.norm()),12.)
            for name,value in model.named_buffers():self.assertTrue(torch.equal(value,buffers[name]));public_buffers[name]=value.clone()
            observed.append(source.calls);return result
        session=Session.synthetic(registry,spec,model_seed=3,sampling_seed=17,noise_seed=19)
        from experiments.node_pdp import sampling
        with patch.object(training,'step',side_effect=checked_step),patch.object(sampling,'sample_once',wraps=sampling.sample_once) as outer:
            result=session.run(graph)
        self.assertEqual(outer.call_count,1)
        self.assertEqual(observed,list(range(1,101)))
        self.assertEqual((result.diagnostics['updates'],result.diagnostics['noise_calls'],result.diagnostics['noise_coordinates']),(100,100,128300))
        self.assertEqual(result.diagnostics['outer_sampling_calls'],1)
        self.assertGreater(result.diagnostics['nonzero_signal_steps'],0)
        self.assertEqual(result.configuration['algorithm'],'rowsum_pdp_c2r_piecewise_v1')
        with self.assertRaises(PermissionError):session.run(graph)

    def test_empty_graph_still100_and_no_production_controls(self):
        from experiments.rowsum_pdp_c2r.research import Session
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import mechanism as prod
        registry,spec,_=self.fixture();before=(base._PROCESS_LEDGER.models,prod._PROCESS_GUARD._used)
        result=Session.synthetic(registry,spec,model_seed=0,sampling_seed=1,noise_seed=2).run(Graph((),(),True))
        self.assertEqual((result.diagnostics['updates'],result.diagnostics['nonzero_signal_steps']),(100,0))
        self.assertEqual((base._PROCESS_LEDGER.models,prod._PROCESS_GUARD._used),before)

    def test_reject_test_sources_in_normal_interface_and_c3_disabled(self):
        from experiments.rowsum_pdp_c2r.research import Session
        registry,spec,graph=self.fixture()
        with self.assertRaisesRegex(PermissionError,'artificial'):Session(registry,spec,model_seed=0)
        session=Session.synthetic(registry,spec,model_seed=0,sampling_seed=0,noise_seed=0)
        session._noise=None
        with self.assertRaisesRegex(PermissionError,'test'):session.run(graph)

    def test_synthetic_bound_and_ones_domain_rejection(self):
        from experiments.rowsum_pdp_c2r.research import Session
        from experiments.rowsum_pdp.config import Spec
        from experiments.node_pdp.registry import Registry
        ids=tuple(range(300));reg=Registry(ids,tuple((v,2) for v in ids),source='public-artificial')
        with self.assertRaises(ValueError):Session.synthetic(reg,Spec(ids),model_seed=0,sampling_seed=0,noise_seed=0)
        model,_=self.api();m=model(0);g=message(4)
        with self.assertRaises(ValueError):m.logits(g,torch.zeros(4,32))
        with self.assertRaises(ValueError):m.logits(message(2,((0,0),)),torch.ones(2,32))

    def test_all_release_paths_reject_candidate(self):
        from experiments.rowsum_pdp_c2r.research import Session
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots import release as slots
        from experiments.node_pdp import release as v1
        from experiments.node_pdp_v2_projected import release as v2
        from experiments.rowsum_pdp_c2r import release
        registry,spec,_=self.fixture()
        result=Session.synthetic(registry,spec,model_seed=0,sampling_seed=0,noise_seed=0).run(Graph((),(),True))
        for mod in (slots,v1,v2,release):
            with self.assertRaises(PermissionError):mod.export(result,permit_idealized=True)

    def test_first_aggregate_hidden_intermediate_logits_and_loss_gradients(self):
        Model,_=self.api()
        cases=[(0,(),True,[]),(1,(),True,[1]),(4,(),True,[1]*4),(4,(),False,[1]*4),
            (2,((0,1),),True,[1,0]),(5,((0,1),(0,2),(1,2),(2,3)),True,[1.5,.5,1,0,0]),
            (6,((0,1),(2,3),(3,2)),True,[1,0,1,1,0,0]),
            (5,((0,1),(1,2)),False,[.5,2,.5,0,0]),(5,((0,1),(2,3)),False,[1,1,1,1,0])]
        maximum=dict(aggregate=0.,hidden=0.,layer2=0.,logits=0.,gradient=0.)
        def compare(a,b,key):
            torch.testing.assert_close(a,b,rtol=RTOL,atol=ATOL)
            if a.numel():maximum[key]=max(maximum[key],float((a-b).abs().max().detach()))
        def grad(p):return torch.zeros_like(p) if p.grad is None else p.grad
        for seed in range(5):
            for changed in (False,True):
                model=Model(seed);old=new_model(seed)
                if changed:
                    with torch.no_grad():model.a.add_(.125);model.b1.sub_(.0625)
                    old.load_state_dict(model.reconstruct())
                for n,edges,directed,expected_q in cases:
                    g=message(n,edges,directed);x=g.ndata['feat'];capture={};new_capture={}
                    hooks=[old.grat.grat1.fc.register_forward_pre_hook(lambda m,args:capture.update(aggregate=args[0].detach().clone())),
                           old.grat.grat1.register_forward_hook(lambda m,args,value:capture.update(hidden=torch.relu(value))),
                           old.grat.grat2.register_forward_hook(lambda m,args,value:capture.update(layer2=torch.relu(value))),
                           model.layer2.register_forward_hook(lambda m,args,value:new_capture.update(layer2=torch.relu(value)))]
                    old.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
                    try:old_logits=old.logits(g,x);new_logits=model.logits(g,x)
                    finally:
                        for h in hooks:h.remove()
                    wanted=torch.tensor(expected_q,dtype=torch.float32)[:,None].expand(n,32)
                    compare(capture['aggregate'],wanted,'aggregate')
                    compare(model.q1(g,x)[:,None].expand(n,32),wanted,'aggregate')
                    compare(capture['hidden'],model.first_hidden(g,x),'hidden')
                    compare(capture['layer2'],new_capture['layer2'],'layer2');compare(old_logits,new_logits,'logits')
                    graph=runtime()[4].Graph(n=n,edges=edges,directed=directed)
                    if not directed:graph=runtime()[4].Graph(n=n,edges=list(edges)+[(v,u) for u,v in edges],directed=True)
                    # Legacy get_adj_mat has a pre-existing 0-node array-shape bug.
                    # No actual SCS/BES item is empty; test the original loss on its
                    # mathematical 0x0 adjacency here, without changing that helper.
                    adjacency=runtime()[3](graph) if n else torch.empty((0,0));loss=runtime()[1](.5)
                    loss(torch.sigmoid(old_logits).squeeze(1),adjacency,50).backward()
                    loss(torch.sigmoid(new_logits).squeeze(1),adjacency,50).backward()
                    compare(grad(old.grat.grat1.fc.weight),grad(model.a)[:,None].expand(32,32),'gradient')
                    compare(grad(old.grat.grat1.fc.bias),grad(model.b1),'gradient')
                    for p in old.grat.grat1.attn_fc.parameters():compare(grad(p),torch.zeros_like(p),'gradient')
                    for layer,original in ((model.layer2,old.grat.grat2),(model.layer3,old.grat.grat3)):
                        for p,o in zip(layer.parameters(),original.parameters()):compare(grad(p),grad(o),'gradient')
        (OUT/'function_gradient_validation.json').write_text(json.dumps(dict(public_initialization_seeds=list(range(5)),cases_per_state=len(cases),parameter_states=['initial','public_a_plus_.125_b1_minus_.0625'],max_absolute_errors=maximum,rtol=RTOL,atol=ATOL),indent=2)+'\n')

    def test_exhaustive_graphs_up_to_three_nodes_first_aggregate(self):
        Model,_=self.api();model=Model(0);old=new_model(0);count=0
        for directed in (True,False):
            for n in range(4):
                arcs=[(u,v) for u in range(n) for v in range(n) if u!=v and (directed or u<v)]
                for bits in itertools.product((False,True),repeat=len(arcs)):
                    edges=[e for e,yes in zip(arcs,bits) if yes];g=message(n,edges,directed);x=g.ndata['feat'];captured=[]
                    h=old.grat.grat1.fc.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach().clone()))
                    with torch.no_grad():old.grat.grat1(g,x)
                    h.remove()
                    torch.testing.assert_close(captured[0],model.q1(g,x)[:,None].expand(n,32),rtol=RTOL,atol=ATOL);count+=1
        self.assertEqual(count,82)

    def test_zero_relu_derivative_in_edgeless_and_empty_domains(self):
        Model,_=self.api()
        for n in (0,1,4):
            old=new_model(0);model=Model(0);g=message(n);x=g.ndata['feat']
            with torch.no_grad():old.grat.grat1.fc.weight.zero_();old.grat.grat1.fc.bias.zero_();model.a.zero_();model.b1.zero_()
            torch.relu(old.grat.grat1(g,x)).sum().backward();model.first_hidden(g,x).sum().backward()
            self.assertTrue(torch.equal(old.grat.grat1.fc.weight.grad,torch.zeros(32,32)))
            self.assertTrue(torch.equal(model.a.grad,torch.zeros(32)));self.assertTrue(torch.equal(model.b1.grad,torch.zeros(32)))

    def test_branch_uses_current_item_not_parent_or_component(self):
        Model,_=self.api();model=Model(0)
        parent=message(5,((0,1),));child=message(3)
        self.assertTrue(torch.equal(model.q1(parent,parent.ndata['feat']),torch.tensor([1.,0.,0.,0.,0.])))
        self.assertTrue(torch.equal(model.q1(child,child.ndata['feat']),torch.ones(3)))

    def test_locality_actual_batches_contributions_and_deletion_sensitivity(self):
        Model,_=self.api();_,spec,graph=self.fixture()
        from experiments.node_dp_slots.adapter import Adapter,Graph
        from experiments.node_dp_slots.config import memberships
        from experiments.rowsum_pdp.training import center
        for actual in (graph,Graph(graph.nodes,((3,7),),True)):
            left=Adapter(spec,actual);right=Adapter(spec,actual.delete(3));unaffected=set(range(4))-set(memberships(3))
            batches=left.batch(),right.batch()
            for j in unaffected:
                self.assertEqual(left.fingerprints()[j],right.fingerprints()[j]);self.assertEqual(batches[0][j],batches[1][j])
            l,ls=center(Model(0),batches[0]);r,rs=center(Model(0),batches[1])
            for j in unaffected:self.assertTrue(torch.equal(ls[j],rs[j]))
            self.assertLessEqual(float((l-r).norm()),12.)

    def test_frozen_scs_bes_extreme_subgraphs_and_empty_containers(self):
        from experiments.sampling import sample_privim_star
        from experiments.node_dp_slots.adapter import Adapter,Graph
        import random
        Model,_=self.api();model=Model(0);old=new_model(0);counts=[]
        for directed in (True,False):
            for n,kind in ((0,'path'),(4,'path'),(20,'path'),(40,'path'),(20,'complete'),(40,'complete')):
                edges=tuple((i,i+1) for i in range(n-1)) if kind=='path' else tuple((u,v) for u in range(n) for v in range(n) if u!=v and (directed or u<v))
                ig=runtime()[4].Graph(n=n,edges=edges,directed=directed)
                result=sample_privim_star(ig,size=40,boundary_divisor=2,sample_rate=1.,restart_probability=.3,walk_length=200,threshold=4,decay=1.,include_boundary_stage=True,rng=random.Random(0))
                if n<20:self.assertEqual(result.node_sets,[])
                for nodes in result.node_sets:
                    self.assertIn(len(nodes),(20,40));child=ig.induced_subgraph(nodes)
                    self.assertGreater(child.ecount(),0)
                    g=runtime()[2](child,'ones',32,directed,False);x=g.ndata['feat']
                    torch.testing.assert_close(model.logits(g,x),old.logits(g,x),rtol=RTOL,atol=ATOL)
                counts.append(dict(directed=directed,nodes=n,kind=kind,stage1=result.stage1_count,stage2=result.stage2_count))
        self.assertGreater(sum(x['stage1'] for x in counts),0);self.assertGreater(sum(x['stage2'] for x in counts),0)
        (OUT/'scs_bes_artificial_summary.json').write_text(json.dumps(counts,indent=2)+'\n')

    def test_original_c2_session_remains_quarantined(self):
        from experiments.rowsum_pdp.research import Session
        registry,spec,graph=self.fixture()
        with self.assertRaisesRegex(PermissionError,'C2 BLOCKED'):
            Session.synthetic(registry,spec,model_seed=0,sampling_seed=0,noise_seed=0).run(graph)

    def test_heterogeneous_outer_draws_once_per_existing_identity(self):
        from experiments.rowsum_pdp_c2r.research import Session
        from experiments.node_pdp.registry import Registry
        from experiments.node_pdp import sampling
        from experiments.node_dp_slots.adapter import Graph
        registry,spec,_=self.fixture()
        registry=Registry(spec.universe,tuple((v,(1,2,4)[v%3]) for v in spec.universe),source='public-artificial-mixed')
        graph=Graph(tuple(range(6)),((0,1),(2,3)),True)
        session=Session.synthetic(registry,spec,model_seed=0,sampling_seed=0,noise_seed=0)
        calls=[];draw=session._sampler.draw
        def observed(p):calls.append(p);return draw(p)
        with patch.object(session._sampler,'draw',side_effect=observed):result=session.run(graph)
        self.assertEqual(calls,[registry.entry(v).retention_probability for v in graph.nodes])
        self.assertEqual(result.diagnostics['outer_sampling_calls'],1)

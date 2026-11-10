"""C3 artificial-only new interface checks, frozen tolerances, no prepared reads."""
import importlib.util
import unittest


class C3EntryTests(unittest.TestCase):
    def test_independent_c3_entry_is_present(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.c3_rowsum'),'independent bounded C3 entry required')

    @staticmethod
    def group_graph(n=40):
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_dp_slots.config import Spec
        from experiments.node_pdp.registry import Registry
        ids=tuple(range(n));graph=Graph(ids,tuple((i,j) for i in range(n) for j in range(n) if i!=j),True)
        registry=Registry(ids,tuple((i,2) for i in ids),source='public artificial ring')
        return registry,Spec(ids),graph

    def test_shared_schedule_exact_original_adapter(self):
        from experiments.c3_rowsum import Group
        from experiments.node_dp_slots.adapter import Adapter
        reg,spec,graph=self.group_graph();group=Group(reg,spec,graph,0,synthetic=True,coin_seed=0,noise_seeds=(11,12,13,14))
        adapter=Adapter(spec,graph)
        self.assertEqual(group._schedule,tuple(tuple(tuple(x) for x in adapter.batch()) for _ in range(100)))
        self.assertTrue(all(group.witness.values()))

    def test_both_100_step_trajectories_match_original_execute(self):
        import torch
        from experiments.c3_rowsum import Group,Model
        from experiments.node_dp_slots import mechanism as base,training
        from experiments.node_pdp_v2_projected import policy
        from experiments.node_dp_slots.noise import TestGaussian as OldTest
        from experiments.rowsum_pdp import training as phi
        from experiments.rowsum_pdp.noise import TestGaussian as PhiTest
        reg,spec,graph=self.group_graph()
        group=Group(reg,spec,graph,0,synthetic=True,coin_seed=0,noise_seeds=(11,12,13,14))
        actual,diag=group.train_next('pdp_v2',0)
        expected=base.new_model(0);layout=policy.Layout.from_model(expected)
        training.execute(spec,graph,OldTest(11),model=expected,step=lambda m,s,z:policy.step(m,s,z,layout),check_slot_bounds=True,check_noise=True)
        for x,y in zip(actual.parameters(),expected.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertEqual((diag['updates'],diag['gaussian_coordinates']),(100,234000))
        # Consume second independent control in preregistered order.
        second,_=group.train_next('pdp_v2',1)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(actual.parameters(),second.parameters())))
        candidate,diag=group.train_next('rowsum_c2r',0)
        expected=Model(0);phi.execute(group.phi_spec,graph,expected,PhiTest(13))
        for x,y in zip(candidate.parameters(),expected.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertEqual((diag['updates'],diag['gaussian_coordinates']),(100,128300))
        last,_=group.train_next('rowsum_c2r',1)
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(candidate.parameters(),last.parameters())))
        with self.assertRaises(PermissionError):group.train_next('pdp_v2',0)

    def test_research_rejects_unvalidated_or_test_randomness(self):
        from experiments.c3_rowsum import Group
        reg,spec,graph=self.group_graph(4)
        with self.assertRaises(PermissionError):Group(reg,spec,graph,0,synthetic=False,preflight={})
        group=Group(reg,spec,graph,0,synthetic=True,coin_seed=0,noise_seeds=(1,2,3,4))
        group.synthetic=False
        with self.assertRaises(PermissionError):group.train_next('pdp_v2',0)
        self.assertEqual(group._sources[0].calls,0)

    def test_public_initialization_layout_and_empty_piecewise(self):
        import torch
        from experiments.c3_rowsum import Model
        from experiments.rowsum_pdp.model import Layout
        from experiments.node_dp_slots import mechanism as base
        import dgl
        for seed in range(5):
            original=base.new_model(seed);candidate=Model(seed)
            for name,value in original.state_dict().items():self.assertTrue(torch.equal(value,candidate.reconstruct()[name]))
            layout=Layout.from_model(candidate);self.assertEqual((layout.total,layout.start,layout.stop),(1283,1185,1217))
            for n in (0,4):
                graph=dgl.graph(([],[]),num_nodes=n);features=torch.ones(n,32)
                torch.testing.assert_close(original.logits(graph,features),candidate.logits(graph,features),rtol=2e-5,atol=2e-5)
                self.assertTrue(torch.equal(candidate.q1(graph,features),torch.ones(n)))

    def test_sparse_evaluation_matches_original_and_candidate_wrapper(self):
        from experiments.c3_rowsum import evaluate,Model
        from experiments.node_dp_slots import mechanism as base
        from experiments.pdp_research_evaluation import prepare_evaluation_graph,evaluate as old_evaluate
        import igraph
        graph=igraph.Graph(n=60,edges=[(i,(i+1)%60) for i in range(60)],directed=True)
        message,coverage=prepare_evaluation_graph(graph);original=base.new_model(2)
        result=evaluate(original,coverage,message,60);legacy=old_evaluate(original,coverage,message,60)
        self.assertEqual((result['spread'],result['coverage']),(legacy['spread'],legacy['coverage']))
        self.assertEqual(evaluate(Model(2),coverage,message,60),result)

    def test_no_export_and_legacy_quarantines_unchanged(self):
        from experiments.c3_rowsum import export
        from experiments.rowsum_pdp import research as old
        from experiments.rowsum_pdp_c2r import research as revised
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import mechanism as prod
        before=base._PROCESS_LEDGER.models,prod._PROCESS_GUARD._used
        for module in (old,revised):
            with self.assertRaises(PermissionError):object.__new__(module.Session).run(None)
        with self.assertRaises(PermissionError):export(object())
        self.assertEqual(before,(base._PROCESS_LEDGER.models,prod._PROCESS_GUARD._used))

    def test_public_result_schema_cannot_mark_incomplete_as_complete(self):
        from experiments.c3_rowsum import trajectory_record
        task=dict(task_id='artificial',status='NOT_RUN',updates=0)
        result=dict(spread=50,coverage=50.)
        row=trajectory_record(task,result,dict(updates=100,gaussian_calls=100))
        self.assertEqual((row['status'],row['updates']),('COMPLETED',100))
        with self.assertRaises(RuntimeError):trajectory_record(task,result,dict(updates=99,gaussian_calls=100))

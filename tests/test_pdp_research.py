"""Synthetic numerical/production-isolation checks for ordinary paper experiments."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import torch


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.node_dp_slots.training'))
        from experiments.node_pdp_research import Session
        return Session

    def fixture(self):
        from experiments.node_dp_slots.config import Spec
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_pdp.registry import Registry
        ids=tuple(range(128));r=Registry(ids,tuple((v,2) for v in ids),source='public-artificial-research')
        return r,Spec(ids),Graph(tuple(range(96)),tuple((v,(v+k)%96) for v in range(96) for k in range(1,13)),True)

    def reference(self,graph,spec,projected):
        # Test-only frozen pre-refactor algorithm; not an alternative research trainer.
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_dp_slots.noise import TestGaussian
        from experiments.node_pdp_v2_projected import policy
        adapter=Adapter(spec,graph);model=base.new_model();source=TestGaussian(19);layout=policy.Layout.from_model(model)
        for _ in range(100):
            signal,_=base.center(model,adapter.batch())
            if projected:policy.step(model,signal,source,layout)
            else:base.update(model.parameters(),signal,source)
        return tuple(float(v) for v in torch.cat([p.detach().reshape(-1) for p in model.parameters()]))

    def test_v1_production_and_research_match_frozen_numeric_loop(self):
        Session=self.api();r,s,g=self.fixture()
        from experiments.node_pdp.mechanism import Engine
        expected=self.reference(g,s,False)
        self.assertEqual(Engine.testing(r,s,sampling_seed=17,noise_seed=19).run(g)._base_run._parameters,expected)
        actual=Session.synthetic(r,s,'node_pdp_v1',sampling_seed=17,noise_seed=19).run(g)
        self.assertEqual(actual.parameters,expected);self.assertEqual(actual.diagnostics['updates'],100)

    def test_v2_production_and_research_match_frozen_numeric_loop(self):
        Session=self.api();r,s,g=self.fixture()
        from experiments.node_pdp_v2_projected.mechanism import Engine
        expected=self.reference(g,s,True)
        self.assertEqual(Engine.testing(r,s,sampling_seed=17,noise_seed=19).run(g)._parameters,expected)
        result=Session.synthetic(r,s,'node_pdp_v2_projected',sampling_seed=17,noise_seed=19).run(g)
        self.assertEqual(result.parameters,expected);self.assertTrue(result.diagnostics['all_projected_weights_nonnegative'])

    def test_research_does_not_touch_consumed_production_controls(self):
        Session=self.api();r,s,_=self.fixture()
        from tests.test_node_pdp import NodePDPTests
        from experiments.node_pdp import mechanism as prod
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.adapter import Graph
        with NodePDPTests().privacy_context():
            prod._PROCESS_GUARD.reserve(base.Engine.privacy(s))
            guard,ledger=prod._PROCESS_GUARD,base._PROCESS_LEDGER
            for _ in range(2):
                result=Session(r,s,'node_pdp_v2_projected').run(Graph((),(),True))
                self.assertEqual(result.diagnostics['noise_coordinates'],234000)
            self.assertIs(guard,prod._PROCESS_GUARD);self.assertIs(ledger,base._PROCESS_LEDGER)
            self.assertTrue(guard._used);self.assertEqual(ledger.models,1)
            with self.assertRaises(PermissionError):prod.Engine.privacy(r,s).run(None)

    def test_research_results_cannot_enter_production_release(self):
        Session=self.api();r,s,_=self.fixture()
        from experiments.node_dp_slots.adapter import Graph
        from experiments.node_pdp import release as v1
        from experiments.node_pdp_v2_projected import release as v2
        result=Session.synthetic(r,s,'node_pdp_v2_projected',sampling_seed=0,noise_seed=0).run(Graph((),(),True))
        for module in [v1,v2]:
            for flag in [False,True]:
                with self.assertRaises(PermissionError):module.export(result,permit_idealized=flag)

    def test_research_rejects_test_sources_and_does_not_rerun_session(self):
        Session=self.api();r,s,_=self.fixture()
        from experiments.node_dp_slots.noise import TestGaussian
        from experiments.node_dp_slots.adapter import Graph
        session=Session(r,s,'node_pdp_v2_projected');session._noise=TestGaussian(0)
        with self.assertRaises(PermissionError):session.run(Graph((),(),True))
        with self.assertRaises(PermissionError):session.run(Graph((),(),True))

    def test_shared_loop_called_and_empty_slots_still_update(self):
        Session=self.api();r,s,_=self.fixture()
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import sampling
        from experiments.node_dp_slots.adapter import Graph
        with patch.object(base,'update',wraps=base.update) as update,patch.object(sampling,'sample_once',wraps=sampling.sample_once) as sample:
            result=Session.synthetic(r,s,'node_pdp_v2_projected',sampling_seed=0,noise_seed=0).run(Graph((),(),True))
        self.assertEqual(sample.call_count,1);self.assertEqual(update.call_count,100)
        self.assertTrue(all(x['positions']==16 and x['zero_gradient_slots']==4 for x in result.diagnostics['steps']))

    def test_no_removed_system_in_active_code(self):
        self.api()
        self.assertFalse(Path('experiments/node_pdp_benchmark').exists())
        self.assertFalse(Path('tests/test_node_pdp_benchmark.py').exists())
        for name in ['experiments/node_pdp_v2_projected/mechanism.py','experiments/node_pdp_research.py','AGENTS.md']:
            self.assertNotIn('benchmark-only',Path(name).read_text())

    def test_evaluation_uses_original_logits_and_outgoing_coverage(self):
        self.api()
        from experiments.pdp_research_evaluation import evaluate
        from experiments.node_dp_slots.mechanism import new_model
        from experiments import node_sampling_utility as u
        import igraph
        model=new_model()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
        graph=igraph.Graph(n=128,edges=[(50,v) for v in range(128) if v!=50],directed=True)
        dg,cg,_=u.prepare_message_graph(graph)
        record=evaluate(model,cg,dg,128)
        self.assertEqual(record['spread'],50);self.assertEqual(record['N_plus'],0)
        self.assertEqual(record['last_zero_aggregate_nodes'],128)
        self.assertEqual(record['selected_zero_outdegree'],50)
        self.assertEqual(record['bias_tie_nodes'],128)

    def test_loader_does_not_infer_public_registry_from_private_graph(self):
        self.api()
        from experiments.run_pdp_research import load_inputs
        with self.assertRaises(ValueError):load_inputs('email',None,432)


if __name__=='__main__':unittest.main()

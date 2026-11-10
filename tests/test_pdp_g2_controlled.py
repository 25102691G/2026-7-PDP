"""Artificial-only controlled-pair checks; no prepared graph training."""
import unittest
from unittest.mock import patch


class G2Tests(unittest.TestCase):
    def api(self):
        from experiments import pdp_g2_controlled
        return pdp_g2_controlled

    def fixture(self):
        from tests.test_pdp_research import ResearchTests
        # Reuse the existing entirely artificial public-domain fixture.
        _,spec,graph=ResearchTests().fixture()
        from experiments.node_pdp.registry import Registry
        registry=Registry(spec.universe,tuple((v,(1,2,4)[v%3]) for v in spec.universe),source='public-artificial-G2')
        return registry,spec,graph

    def test_exact_twenty_four_cells_twelve_outer_groups(self):
        tasks=self.api().task_matrix();self.assertEqual(len(tasks),24)
        groups={}
        for t in tasks:groups.setdefault((t['dataset'],t['model_seed'],t['outer_index']),[]).append(t['gaussian_replicate'])
        self.assertEqual(len(groups),12)
        self.assertTrue(all(sorted(v)==[0,1] for v in groups.values()))
        self.assertEqual({t['model_seed'] for t in tasks},{1,2})

    def test_pair_matches_existing_kernel_and_does_not_charge_ledger(self):
        import torch
        torch.set_num_threads(1)
        api=self.api();registry,spec,graph=self.fixture()
        from experiments.node_pdp_research import Session
        from experiments.node_dp_slots.accounting import Ledger
        from experiments.node_pdp import sampling
        with patch.object(Ledger,'charge',side_effect=AssertionError('no formal charge')):
            with patch.object(sampling,'sample_once',wraps=sampling.sample_once) as draw:
                pair=api.ControlledPair.synthetic(registry,spec,graph,model_seed=0,sampling_seed=17,noise_seeds=(19,20))
                first=pair.train_next();second=pair.train_next()
                self.assertEqual(draw.call_count,1)
            expected=Session.synthetic(registry,spec,'node_pdp_v2_projected',sampling_seed=17,noise_seed=19).run(graph)
        self.assertEqual(first.parameters,expected.parameters)
        self.assertNotEqual(first.parameters,second.parameters)
        self.assertEqual(first.initial_hash,second.initial_hash)
        self.assertEqual(second.diagnostics['paired_batch_steps_equal'],100)
        self.assertTrue(second.diagnostics['paired_caches_equal'])
        self.assertTrue(second.diagnostics['distinct_noise_source'])
        for result in (first,second):
            self.assertEqual(result.diagnostics['updates'],100)
            self.assertEqual(result.diagnostics['noise_coordinates'],234000)
            self.assertTrue(result.diagnostics['all_projected_weights_nonnegative'])
        import json
        serialized=json.dumps(second.diagnostics)
        for forbidden in ('noise_seed','sampling_seed','_cache_signature','_batch_signatures','random_state'):
            self.assertNotIn(forbidden,serialized)
        self.assertEqual(set(vars(second)),{'model','parameters','initial_hash','diagnostics','sampling_diagnostics'})
        with self.assertRaises(PermissionError):pair.train_next()

    def test_single_forward_metrics_match_existing_evaluator_on_artificial_graph(self):
        from experiments.node_dp_slots import mechanism as base
        from experiments.pdp_research_evaluation import prepare_evaluation_graph,evaluate
        from experiments.run_pdp_g2 import evaluate_once
        import igraph
        graph=igraph.Graph(n=96,edges=[(v,(v+k)%96) for v in range(96) for k in range(1,13)],directed=True)
        message,coverage=prepare_evaluation_graph(graph);model=base.new_model(1)
        calls=[];handle=model.grat.register_forward_hook(lambda *args:calls.append(1))
        actual=evaluate_once(model,coverage,message,96)
        handle.remove();expected=evaluate(model,coverage,message,96)
        self.assertEqual(len(calls),1)
        for key in expected:self.assertEqual(actual[key],expected[key])
        self.assertIsNone(actual['spearman_nonzero_degree'])
        self.assertEqual(actual['A'],650)

    def test_real_mode_rejects_test_sources_before_training(self):
        api=self.api();r,s,g=self.fixture()
        pair=api.ControlledPair.synthetic(r,s,g,model_seed=0,sampling_seed=1,noise_seeds=(2,3))
        pair.purpose='research'
        with self.assertRaises(PermissionError):pair.train_next()

    def test_variance_components_keep_negative_outer_estimate(self):
        api=self.api();v=api.variance_components([[0,2],[0,2]])
        self.assertEqual(v['gaussian_residual_variance'],2)
        self.assertEqual(v['outer_variance_raw'],-1)
        self.assertEqual(v['outer_variance_nonnegative_proxy'],0)


if __name__=='__main__':unittest.main()

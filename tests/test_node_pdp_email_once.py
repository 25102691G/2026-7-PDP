"""One-run harness checks: artificial identities/graphs ONLY."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class EmailOnceTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.node_pdp_email_once'))
        from experiments import node_pdp_email_once
        return node_pdp_email_once

    def fixture(self):
        from experiments.node_pdp.registry import Registry
        from experiments.node_dp_slots.config import Spec
        from experiments.node_dp_slots.adapter import Graph
        ids=tuple(range(128));reg=Registry(ids,tuple((v,2) for v in ids),source='public-artificial-harness')
        graph=Graph(tuple(range(96)),tuple((v,(v+k)%96) for v in range(96) for k in range(1,13)),True)
        return reg,Spec(ids),graph

    def privacy_context(self):
        from tests.test_node_pdp import NodePDPTests
        return NodePDPTests().privacy_context()

    def test_attempt_file_is_exclusive_and_survives_failure(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);api.reserve_attempt(root)
            with self.assertRaises(FileExistsError):api.reserve_attempt(root)
            record=json.loads((root/'training_attempt.json').read_text())
            self.assertEqual(record['maximum_real_training_attempts'],1)

    def test_audited_production_path_runs_exactly_once_and_100_steps(self):
        api=self.api();reg,spec,graph=self.fixture()
        with self.privacy_context():
            result,audit=api.audited_run(reg,spec,graph)
            self.assertEqual(result._base_run._mode,'privacy')
            self.assertEqual(audit['outer_sample_calls'],1)
            self.assertEqual(audit['induced_graph_calls'],1)
            self.assertEqual(audit['adapter_calls'],1)
            self.assertEqual(audit['updates'],100)
            self.assertEqual(audit['noise_calls'],100)
            self.assertEqual(audit['noise_coordinates'],234000)
            self.assertEqual(audit['double_clip_calls'],100)
            self.assertTrue(audit['all_parameters_finite'])
            self.assertEqual(len(audit['steps']),100)
            self.assertTrue(any(s['effective_positions'] for s in audit['steps']))
            self.assertTrue(all(s['positions']==16 for s in audit['steps']))
            self.assertEqual(set(audit['source_types']),{'SecretBernoulli','SecretGaussian'})
            self.assertNotIn('noise_vectors',audit)
            self.assertNotIn('retained_ids',audit)
            from experiments.node_pdp.release import export
            with self.assertRaises(PermissionError):export(result)

    def test_nonfinite_noise_aborts_without_retry(self):
        api=self.api();reg,spec,graph=self.fixture()
        import torch
        from experiments.node_dp_slots.noise import SecretGaussian
        from experiments.node_pdp.mechanism import ExecutionGap
        with self.privacy_context():
            with patch.object(SecretGaussian,'draw',return_value=torch.full((2340,),float('inf'))):
                with self.assertRaises(ExecutionGap):api.audited_run(reg,spec,graph)
            with self.assertRaises(PermissionError):api.audited_run(reg,spec,graph)

    def test_testing_sources_cannot_enter_audited_real_path(self):
        api=self.api();reg,spec,graph=self.fixture()
        from experiments.node_pdp.mechanism import Engine
        testing=Engine.testing(reg,spec,sampling_seed=0,noise_seed=0)
        with patch.object(Engine,'privacy',return_value=testing):
            with self.assertRaises(PermissionError):api.audited_run(reg,spec,graph)

    def test_nonfinite_training_forward_stops_instead_of_silent_continuation(self):
        api=self.api();reg,spec,graph=self.fixture()
        import torch
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp.mechanism import ExecutionGap
        model=base.new_model()
        model.forward=lambda g,x:torch.full((x.shape[0],1),float('nan'))
        with self.privacy_context(),patch.object(base,'new_model',return_value=model):
            with self.assertRaises(ExecutionGap):api.audited_run(reg,spec,graph)

    def test_parameter_reconstruction_is_exact_without_updates(self):
        api=self.api()
        from experiments.node_dp_slots.mechanism import new_model
        import torch
        initial=new_model();flat=tuple(float(x) for x in torch.cat([p.detach().reshape(-1) for p in initial.parameters()]))
        restored=api.restore_model(flat)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(initial.parameters(),restored.parameters())))

    def test_evaluation_uses_logits_and_original_one_step_protocol(self):
        api=self.api()
        from experiments.node_dp_slots.mechanism import new_model
        import igraph
        graph=igraph.Graph(n=64,edges=[(i,(i+1)%64) for i in range(64)],directed=True)
        result=api.evaluate(new_model(),graph,'artificial')
        self.assertGreaterEqual(result['influence_spread'],50)
        self.assertLessEqual(result['influence_spread'],64)
        self.assertEqual(len(result['top50_local_ids']),50)
        self.assertAlmostEqual(result['coverage_ratio_percent'],100*result['influence_spread']/432)
        self.assertTrue(result['finite_outputs'])


if __name__=='__main__':unittest.main()

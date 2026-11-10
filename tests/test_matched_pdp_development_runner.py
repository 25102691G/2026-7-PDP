"""New prepared-runner behavior tested with artificial inputs only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.run_matched_pdp_frozen_backbone_research'))
        from experiments import run_matched_pdp_frozen_backbone_research as runner
        return runner

    def fixture(self, root):
        from experiments.node_pdp.registry import Registry
        from experiments.node_dp_slots.config import Spec
        from experiments.node_dp_slots.adapter import Graph
        from experiments.matched_pdp_frozen_backbone.layout import new_model
        from experiments.node_sampling_utility import state_hash
        ids=(0,1,2)
        reg=Registry(ids,((0,1),(1,2),(2,4)),source='public-artificial-runner')
        inputs=(reg,Spec(ids),Graph(ids,((0,1),(1,2)),True),None,None,tuple(range(50)),{},
                dict(train_test_disjoint=True,registry_fingerprint=reg.fingerprint,domain_scope='artificial',celf_reference=432))
        task=dict(task_id='artificial',dataset='email',model_seed=2,partition_seed=0,algorithm='matched_pdp_frozen_backbone_v1',
            T=100,head_dimension=33,full_dimension=2340,sigma_sum=260,reference_spread=432,
            output=str(root/'outputs/artificial'),registry='unused',registry_fingerprint=reg.fingerprint,
            input_sha256={},source_sha256={},initial_model_sha256=state_hash(new_model(2)),max_wall_minutes=60,max_rss_gib=8)
        return inputs,task

    def test_runner_matches_unmodified_session_no_retry_and_restricted_outputs(self):
        runner=self.api()
        from experiments.matched_pdp_frozen_backbone.research import Session
        from experiments.matched_pdp_frozen_backbone.config import Spec
        def synthetic(reg,spec,model_seed):
            return Session.synthetic(reg,spec,model_seed=model_seed,sampling_seed=8,noise_seed=9)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs,task=self.fixture(root)
            expected=synthetic(inputs[0],Spec(inputs[0].universe,True),2).run(inputs[2])
            with patch.object(runner,'ROOT',root),patch.object(runner,'load_inputs',return_value=inputs) as load,patch.object(runner,'make_session',side_effect=synthetic),patch.object(runner,'evaluate',return_value=dict(spread=50,coverage=100*50/432,topk_local_ids=list(range(50)))):
                runner.run(task)
                with self.assertRaises(FileExistsError):runner.run(task)
            self.assertEqual(load.call_count,1)
            dest=root/'outputs/artificial/restricted'
            model=json.loads((dest/'model.json').read_text())
            self.assertEqual(model['parameters'],list(expected.parameters))
            done=json.loads((dest/'completion.json').read_text())
            self.assertEqual(done['updates'],100);self.assertEqual(done['actual_counts']['gaussian_vectors'],100)
            self.assertEqual(done['actual_counts']['noisy_updates'],100)
            self.assertEqual(done['noise_coordinates'],3300)
            self.assertTrue(done['frozen_backbone_unchanged']);self.assertTrue(done['production_controls_unchanged'])
            for f in dest.iterdir():
                self.assertEqual(f.stat().st_mode & 0o777,0o600)
                text=f.read_text()
                for token in ('noise_seed','retained_mask','random_bytes','noise_vector','rng_state'):
                    self.assertNotIn(token,text)

    def test_failure_is_consumed_and_no_secret_exception_text_is_saved(self):
        runner=self.api()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,task=self.fixture(root)
            with patch.object(runner,'ROOT',root),patch.object(runner,'load_inputs',side_effect=RuntimeError('SECRET_PRIVATE_DETAIL')):
                with self.assertRaises(RuntimeError):runner.run(task)
                with self.assertRaises(FileExistsError):runner.run(task)
            f=root/'outputs/artificial/restricted/failure.json'
            self.assertNotIn('SECRET_PRIVATE_DETAIL',f.read_text())
            self.assertEqual(json.loads(f.read_text())['status'],'FAILED_NO_RETRY')

    def test_resource_abort_terminates_even_if_failure_marker_cannot_be_written(self):
        runner=self.api()
        self.assertTrue(hasattr(runner,'abort_resource'))
        with tempfile.TemporaryDirectory() as d:
            with patch.object(runner,'save',side_effect=RuntimeError('disk full')),patch.object(runner.os,'_exit',side_effect=SystemExit(70)) as terminate:
                with self.assertRaises(SystemExit) as caught:runner.abort_resource(Path(d),runner.time.monotonic())
            self.assertEqual(caught.exception.code,70);terminate.assert_called_once_with(70)

    def test_resource_guards_cover_wall_rss_disk_and_prewrite(self):
        runner=self.api()
        self.assertTrue(hasattr(runner,'resource_violation'))
        task=dict(max_wall_minutes=60,max_rss_gib=8)
        self.assertFalse(runner.resource_violation(task,1,1,1))
        self.assertTrue(runner.resource_violation(task,3601,1,1))
        self.assertTrue(runner.resource_violation(task,1,8*2**30+1,1))
        self.assertTrue(runner.resource_violation(task,1,1,2*2**30+1))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'large.json'
            with self.assertRaises(RuntimeError):runner.save(path,{'text':'too large'},budget_roots=(Path(d),),max_bytes=5)
            self.assertFalse(path.exists())

    def test_original_evaluator_and_default_scientific_source_on_artificial_graph(self):
        runner=self.api()
        import igraph
        from experiments.pdp_research_evaluation import prepare_evaluation_graph
        path=Path(__file__).resolve().parents[1]/'docs/pdp_research/new_matched_pdp_development/workflow.py'
        spec=importlib.util.spec_from_file_location('matched_validation',path)
        workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs,task=self.fixture(root)
            graph=igraph.Graph.Ring(64,directed=True)
            message,coverage=prepare_evaluation_graph(graph)
            inputs=(*inputs[:3],coverage,message,tuple(range(100,164)),*inputs[6:])
            with patch.object(runner,'ROOT',root),patch.object(runner,'load_inputs',return_value=inputs):
                runner.run(task)
            m=workflow.validate_completed(task)
            self.assertGreaterEqual(m['spread'],50)
            self.assertEqual(m['updates'],100)
            dest=Path(task['output'])/'restricted/completion.json'
            c=json.loads(dest.read_text());c['noise_coordinates']=3301;dest.write_text(json.dumps(c));dest.chmod(0o600)
            with self.assertRaises(RuntimeError):workflow.validate_completed(task)

    def test_bad_protocol_rejected_before_output_or_graph_access(self):
        runner=self.api()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,task=self.fixture(root);task['sigma_sum']=259
            with patch.object(runner,'ROOT',root),patch.object(runner,'load_inputs',side_effect=AssertionError('no graph access')):
                with self.assertRaises(ValueError):runner.run(task)
            self.assertFalse(Path(task['output']).exists())

    def test_changed_source_and_input_hashes_block_without_training(self):
        runner=self.api()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);_,task=self.fixture(root)
            f=root/'source.py';f.write_text('changed');task['source_sha256']={'source.py':'0'*64}
            with patch.object(runner,'ROOT',root),patch.object(runner,'load_inputs',side_effect=AssertionError('no graph access')):
                with self.assertRaises(ValueError):runner.run(task)
            self.assertFalse(Path(task['output']).exists())


if __name__=='__main__':unittest.main()

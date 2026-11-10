"""Artificial-only checks for declared prepared domains and sparse evaluation."""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import igraph
import torch


class PreparedResearchTests(unittest.TestCase):
    def test_registry_uses_all_stable_ids_and_is_order_independent(self):
        from experiments import run_pdp_research as runner
        self.assertTrue(hasattr(runner, 'registry_from_prepared_mapping'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'node_mapping.csv'
            rows = ['101,train,0', '9000,test,0', '500,train,1',
                    '70000,test,1', '12000,train,2']
            path.write_text('original_id,split,local_id\n' + '\n'.join(rows) + '\n')
            first = runner.registry_from_prepared_mapping(path, public_seed=0)
            path.write_text('original_id,split,local_id\n' + '\n'.join(reversed(rows)) + '\n')
            second = runner.registry_from_prepared_mapping(path, public_seed=0)
            self.assertEqual(first.universe, (101, 500, 9000, 12000, 70000))
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual([sum(e == k for _, e in first.budgets) for k in (1, 2, 4)], [1, 2, 2])

    def test_duplicate_identity_across_splits_rejected(self):
        from experiments import run_pdp_research as runner
        self.assertTrue(hasattr(runner, 'registry_from_prepared_mapping'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'node_mapping.csv'
            path.write_text('original_id,split,local_id\n101,train,0\n101,test,0\n')
            with self.assertRaises(ValueError):
                runner.registry_from_prepared_mapping(path, public_seed=0)

    def test_partial_registry_rejected_before_training_graph_loading(self):
        from experiments import run_pdp_research as runner
        from experiments.node_pdp.registry import Registry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / 'toy'
            prepared.mkdir()
            (prepared / 'metadata.json').write_text(json.dumps(dict(train_ratio=.5, directed=True)))
            (prepared / 'node_mapping.csv').write_text('original_id,split,local_id\n101,train,0\n9000,test,0\n')
            registry = Registry((101,), ((101, 2),), source='artificial-declared-domain')
            path = root / 'registry.json'
            path.write_text(json.dumps(dict(registry=registry.public(), fingerprint=registry.fingerprint,
                                           domain_scope='fixed-prepared-mapping')))
            # There are deliberately no graph files: a train-only registry must
            # fail domain validation rather than start reading those files.
            with patch.object(runner, 'PREPARED', root):
                with self.assertRaisesRegex(ValueError, 'complete declared prepared domain'):
                    runner.load_inputs('toy', path, None)

    def test_sparse_message_preparation_never_requests_dense_adjacency(self):
        from experiments import pdp_research_evaluation as evaluation
        from experiments import node_sampling_utility as u
        self.assertTrue(hasattr(evaluation, 'prepare_evaluation_graph'))
        rt = u.runtime()
        for directed in (True, False):
            graph = igraph.Graph(n=6, edges=[(0, 1), (1, 2), (3, 1)], directed=directed)
            with patch.object(rt['graph_util'], 'get_adj_mat', side_effect=AssertionError('dense allocation forbidden')):
                message, coverage = evaluation.prepare_evaluation_graph(graph)
            u.check_message_graph(graph, message, coverage)
            self.assertEqual(message.num_nodes(), 6)

    def test_set_coverage_equals_original_dense_definition_for_all_small_seed_sets(self):
        from experiments import pdp_research_evaluation as evaluation
        from experiments import node_sampling_utility as u
        from baselines.heuristics import get_influence
        self.assertTrue(hasattr(evaluation, 'prepare_evaluation_graph'))
        rt = u.runtime()
        for directed in (True, False):
            graph = igraph.Graph(n=6, edges=[(0, 1), (1, 2), (3, 1), (2, 2)], directed=directed)
            _, coverage = evaluation.prepare_evaluation_graph(graph)
            adjacency = rt['graph_util'].get_adj_mat(coverage)
            for size in range(7):
                for seeds in itertools.combinations(range(6), size):
                    dense_spread = int((adjacency[list(seeds)].sum(0) > 0).sum())
                    self.assertEqual(get_influence(coverage, seeds), dense_spread)
                    self.assertEqual(get_influence(graph, seeds), dense_spread)

    def test_missing_celf_reference_produces_na_coverage(self):
        from experiments import pdp_research_evaluation as evaluation
        from experiments.node_dp_slots.mechanism import new_model
        self.assertTrue(hasattr(evaluation, 'prepare_evaluation_graph'))
        model = new_model()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        graph = igraph.Graph(n=64, edges=[(50, v) for v in range(64) if v != 50], directed=True)
        message, coverage = evaluation.prepare_evaluation_graph(graph)
        record = evaluation.evaluate(model, coverage, message, None)
        self.assertEqual(record['spread'], 50)
        self.assertIsNone(record['coverage'])
        self.assertIsNone(record['reference_spread'])


if __name__ == '__main__':
    unittest.main()

"""Structural feasibility tests; none certify differential privacy."""

import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

import igraph

from experiments import sampling


class NodeSamplingStructureTests(unittest.TestCase):
    def api(self):
        # An assertion gives a useful red test before the independent script exists.
        import importlib.util
        self.assertIsNotNone(
            importlib.util.find_spec("experiments.node_sampling_structure"),
            "independent node-structure experiment has not been implemented",
        )
        from experiments import node_sampling_structure
        return node_sampling_structure

    def test_probabilities_saturate_and_handle_zero_and_large_budgets(self):
        api = self.api()
        values = api.retention_probabilities([0, 1, 2, 4], 2)
        self.assertEqual(values[0], 0)
        self.assertAlmostEqual(values[1], math.expm1(1) / math.expm1(2))
        self.assertEqual(values[2:], [1.0, 1.0])
        self.assertEqual(api.retention_probabilities([1000], 1000), [1.0])
        for budgets, base in [([-1], 2), ([math.nan], 2), ([1], 0)]:
            with self.assertRaises(ValueError):
                api.retention_probabilities(budgets, base)

    def test_budget_assignment_has_fixed_quotas_and_ignores_topology(self):
        api = self.api()
        budgets = api.assign_budgets(462, seed=7)
        self.assertEqual([budgets.count(x) for x in (1, 2, 4)], [92, 185, 185])
        self.assertEqual(budgets, api.assign_budgets(462, seed=7))
        self.assertNotEqual(budgets, api.assign_budgets(462, seed=8))
        self.assertEqual(api.assign_budgets(0, seed=7), [])

    def test_induction_preserves_original_ids_isolates_and_direction(self):
        api = self.api()
        for directed in (True, False):
            graph = igraph.Graph(n=5, edges=[(0, 1), (1, 2)], directed=directed)
            graph.vs["original_id"] = [100, 200, 300, 400, 500]
            graph.vs["prepared_local_id"] = list(range(5))
            induced, mapping = api.induce_nodes(graph, [4, 1, 0])
            self.assertEqual(mapping, {0: 0, 1: 1, 4: 2})
            self.assertEqual(induced.vs["original_id"], [100, 200, 500])
            self.assertEqual(induced.vcount(), 3)
            self.assertEqual(induced.ecount(), 1)
            self.assertEqual(induced.degree(2), 0)
            self.assertEqual(induced.is_directed(), directed)

    def test_bernoulli_is_one_pass_with_no_top_up(self):
        api = self.api()
        class Coins:
            calls = 0
            def random(self):
                self.calls += 1
                return 0.9
        coins = Coins()
        self.assertEqual(api.bernoulli_nodes([0.1] * 5, coins), [])
        self.assertEqual(coins.calls, 5)

    def test_coupled_outer_draws_are_nested_as_base_budget_increases(self):
        api = self.api()
        budgets = api.assign_budgets(100, seed=3)
        selected = [set(api.bernoulli_nodes(
            api.retention_probabilities(budgets, base), random.Random(11),
        )) for base in (2, 3, 4)]
        self.assertLessEqual(selected[2], selected[1])
        self.assertLessEqual(selected[1], selected[0])

    def test_profiler_is_restored_when_sampler_rejects_parameters(self):
        api = self.api()
        previous = sys.getprofile()
        with self.assertRaises(ValueError):
            api.observe_extraction(igraph.Graph(n=0), size=4, threshold=0, seed=0)
        self.assertIs(sys.getprofile(), previous)

    def test_empty_graph_and_failed_walks_have_explicit_denominators(self):
        api = self.api()
        for n in (0, 5):
            graph = igraph.Graph(n=n, directed=True)
            metrics = api.graph_structure(graph, scs_size=4, bes_size=2)
            self.assertEqual(metrics["isolated_nodes"], n)
            observed = api.observe_extraction(graph, size=4, threshold=2, seed=0)
            self.assertEqual(observed["result"].node_sets, [])
            self.assertEqual(observed["walk_successes"], 0)
            self.assertEqual(observed["walk_attempts"], 2 * n)
            self.assertEqual(observed["rwr_success_rate"], 0.0 if n else None)
            self.assertEqual(metrics["largest_wcc_fraction"], 1.0 / n if n else 0.0)

    def test_observation_does_not_change_sampler_or_rng(self):
        api = self.api()
        graph = igraph.Graph.Ring(30, directed=True, mutual=True)
        expected = sampling.sample_privim_star(
            graph, size=4, boundary_divisor=2, sample_rate=1.0,
            restart_probability=0.3, walk_length=200, threshold=3,
            decay=1.0, include_boundary_stage=True, rng=random.Random(9),
        )
        previous = sys.getprofile()
        observed = api.observe_extraction(
            graph, size=4, threshold=3, seed=9, sample_rate=1.0,
        )
        self.assertIs(sys.getprofile(), previous)
        self.assertEqual(observed["result"], expected)
        self.assertEqual(observed["walk_successes"], len(expected.node_sets))
        self.assertEqual(observed["scs_successes"], expected.stage1_count)
        self.assertEqual(observed["bes_successes"], expected.stage2_count)
        self.assertLessEqual(max(expected.frequencies), 3)

    def test_expected_edges_treat_self_loop_as_one_node(self):
        api = self.api()
        graph = igraph.Graph(n=2, edges=[(0, 0), (0, 1)], directed=True)
        node_rate, edge_rate = api.expected_retention(graph, [0.2, 0.5])
        self.assertAlmostEqual(node_rate, 0.35)
        self.assertAlmostEqual(edge_rate, (0.2 + 0.2 * 0.5) / 2)

    def test_tiny_end_to_end_outputs_and_no_overwrite(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared" / "email"
            prepared.mkdir(parents=True)
            metadata = dict(
                directed=True, original_nodes=20, train_nodes=10,
                train_assigned_nodes=10, train_ratio=0.5,
            )
            (prepared / "metadata.json").write_text(json.dumps(metadata))
            (prepared / "email_train_edges.txt").write_text("0 1\n1 2\n2 0\n")
            (prepared / "train_nodes.csv").write_text(
                "original_id,local_id\n" + "".join(f"{100+i},{i}\n" for i in range(10))
            )
            output = root / "runs" / "new-run"
            rows = api.run_experiment(
                prepared_root=root / "prepared", output=output,
                datasets=["email"], base_epsilons=[2], seeds=[0],
            )
            self.assertEqual(len(rows), 3)
            personal = next(r for r in rows if r["arm"] == "personalized")
            uniform = next(r for r in rows if r["arm"] == "uniform_matched")
            self.assertEqual(personal["nodes"], uniform["nodes"])
            for name in ("records.csv", "summary.csv", "report.md", "manifest.json"):
                self.assertTrue((output / name).is_file(), name)
            self.assertIn("不构成", (output / "report.md").read_text())
            with self.assertRaises(FileExistsError):
                api.run_experiment(
                    prepared_root=root / "prepared", output=output,
                    datasets=["email"], base_epsilons=[2], seeds=[0],
                )


if __name__ == "__main__":
    unittest.main()

"""No-noise utility runner tests; these are not privacy certification tests."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
STRUCTURE = ROOT / "outputs/structure_sampling/personalized-node-20261030-v2"


class NodeSamplingUtilityTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(
            importlib.util.find_spec("experiments.node_sampling_utility"),
            "independent no-noise utility runner has not been implemented",
        )
        from experiments import node_sampling_utility
        return node_sampling_utility

    def test_loads_saved_email_cases_without_resampling(self):
        api = self.api()
        cases = [api.load_saved_case(STRUCTURE, ROOT / "dataset/prepared", arm, 0)
                 for arm in api.ARMS]
        self.assertEqual([c["graph"].vcount() for c in cases], [462, 390, 390])
        self.assertEqual([c["graph"].ecount() for c in cases], [6243, 4634, 4223])
        self.assertEqual([len(c["node_sets"]) for c in cases], [35, 28, 27])
        self.assertTrue(all(c["graph"].is_directed() for c in cases))
        for case in cases:
            self.assertEqual(len(case["original_ids"]), case["graph"].vcount())
            self.assertEqual(len(set(case["original_ids"])), case["graph"].vcount())

    def test_paired_coverage_differences_are_percentage_points(self):
        api = self.api()
        rows = [dict(seed=seed, arm=arm, coverage_ratio_percent=coverage)
                for seed, values in [(0, (80, 75, 83)), (1, (90, 94, 88))]
                for arm, coverage in zip(api.ARMS, values)]
        differences = api.paired_differences(rows)
        self.assertEqual([r["delta_vs_original_pp"] for r in differences], [-5, 3, 4, -2])
        self.assertEqual([r["personalized_minus_uniform_pp"] for r in differences], [-8, -8, 6, 6])

    def test_smoke_gate_rejects_bad_outputs_or_mismatched_initialization(self):
        api = self.api()
        rows = [dict(seed=0, arm=arm, updates=100, finite_outputs=True,
                     parameter_shapes={"w": [2, 3]}, initial_state_sha256="same",
                     selected_seed_count=50, unique_seed_count=50,
                     influence_spread=200, coverage_ratio_percent=80,
                     test_graph_sha256="same-test", saved_subgraphs_verified=True)
                for arm in api.ARMS]
        api.validate_smoke(rows)
        for key, value in [("updates", 99), ("finite_outputs", False),
                           ("initial_state_sha256", "other"),
                           ("coverage_ratio_percent", float("nan"))]:
            broken = [dict(row) for row in rows]
            broken[1][key] = value
            with self.assertRaises(ValueError, msg=key):
                api.validate_smoke(broken)

    def test_reversed_message_edges_and_features_match_cover_graph(self):
        api = self.api()
        import igraph
        for directed in (True, False):
            graph = igraph.Graph(n=4, edges=[(0, 1), (1, 2)], directed=directed)
            message_graph, coverage_graph, adjacency = api.prepare_message_graph(graph)
            self.assertEqual(message_graph.num_nodes(), 4)
            self.assertEqual(tuple(message_graph.ndata["feat"].shape), (4, 32))
            self.assertTrue((message_graph.ndata["feat"] == 1).all().item())
            self.assertEqual(adjacency[0, 1].item(), 1)
            self.assertEqual(adjacency[1, 0].item(), 0 if directed else 1)
            self.assertEqual(coverage_graph.vcount(), 4)

    def test_models_have_identical_shapes_and_initial_tensors_within_seed(self):
        api = self.api()
        models = [api.new_model(0) for _ in api.ARMS]
        self.assertEqual(len({api.state_hash(m) for m in models}), 1)
        self.assertEqual(sum(p.numel() for p in models[0].parameters()), 2340)
        self.assertNotEqual(api.state_hash(models[0]), api.state_hash(api.new_model(1)))

    def test_internal_files_restricted_and_existing_output_refused(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "new"
            internal = api.create_output(directory)
            self.assertEqual(internal.stat().st_mode & 0o777, 0o700)
            with self.assertRaises(FileExistsError):
                api.create_output(directory)


if __name__ == "__main__":
    unittest.main()

"""Structural/locality tests only: no model, gradients, or privacy certification."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import unittest

import igraph


class StableSlotTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.stable_slot_structure"))
        from experiments import stable_slot_structure
        return stable_slot_structure

    def graph(self, directed=True):
        g = igraph.Graph(n=24, edges=[(i, (i + 1) % 24) for i in range(24)] +
                          [(i, (i + 3) % 24) for i in range(24)], directed=directed)
        g.vs["original_id"] = [1000 + 13 * i for i in range(24)]
        g.vs["prepared_local_id"] = list(range(24))
        return g

    def test_membership_uses_stable_id_not_local_position(self):
        api = self.api()
        for seed in range(5):
            for node in (0, 7, 1000, 999999999999):
                slots = api.memberships(node, seed)
                self.assertEqual(len(slots), 2)
                self.assertIn(slots[0], (0, 1))
                self.assertIn(slots[1], (2, 3))
                self.assertEqual(slots, api.memberships(node, seed))
        g = self.graph()
        before = api.build_slots(g, 2)
        permuted = g.permute_vertices(list(reversed(range(g.vcount()))))
        self.assertEqual([api.input_payload(x) for x in before],
                         [api.input_payload(x) for x in api.build_slots(permuted, 2)])

    def test_empty_slots_and_quota_are_retained(self):
        api = self.api()
        g = igraph.Graph(n=0, directed=True)
        g.vs["original_id"] = []
        slots = api.build_slots(g, 0)
        self.assertEqual(len(slots), 4)
        caches = [api.extract_slot(x, size=4, threshold=4, seed=i) for i, x in enumerate(slots)]
        self.assertEqual([x["metrics"]["zero_padding"] for x in caches], [4] * 4)
        self.assertEqual([x["metrics"]["subgraphs"] for x in caches], [0] * 4)
        self.assertTrue(all(x["metrics"]["rwr_success_rate"] is None for x in caches))
        self.assertTrue(all(x["cache"]["rng_draw_count"] == 0 for x in caches))

    def test_trace_matches_standard_random_and_observation_preserves_sampler(self):
        api = self.api()
        traced = api.TraceRandom(19)
        plain = random.Random(19)
        for _ in range(20):
            self.assertEqual(traced.random(), plain.random())
            self.assertEqual(traced.choice(list(range(7))), plain.choice(list(range(7))))
        self.assertEqual(traced.getstate(), plain.getstate())
        g = self.graph()
        observed = api.extract_slot(g, size=4, threshold=4, seed=19)
        from experiments import sampling
        rng = random.Random(19)
        result = sampling.sample_privim_star(
            g, size=4, boundary_divisor=2, sample_rate=1., restart_probability=.3,
            walk_length=200, threshold=4, decay=1., include_boundary_stage=True, rng=rng)
        self.assertEqual(observed["cache"]["node_sets"], result.node_sets)
        self.assertEqual(observed["cache"]["frequencies"], result.frequencies)
        self.assertEqual(json.loads(json.dumps(rng.getstate())), observed["cache"]["rng_final_state"])
        self.assertLessEqual(max(result.frequencies), 4)

    def test_isolated_nodes_survive_partition_without_retry_or_top_up(self):
        api = self.api()
        g = igraph.Graph(n=5, directed=True)
        g.vs["original_id"] = [10, 20, 30, 40, 50]
        slots = api.build_slots(g, 0)
        from collections import Counter
        self.assertEqual(Counter(node for slot in slots for node in slot.vs["original_id"]),
                         Counter({node: 2 for node in g.vs["original_id"]}))
        for j, slot in enumerate(slots):
            data = api.extract_slot(slot, size=4, threshold=4, seed=j)
            self.assertEqual(data["metrics"]["walk_attempts"], 2 * slot.vcount())
            self.assertEqual(data["metrics"]["zero_padding"], 4)
            self.assertEqual(data["cache"]["frequencies"], [0] * slot.vcount())

    def test_all_single_deletions_preserve_unaffected_slots_and_rng(self):
        api = self.api()
        for directed in (True, False):
            g = self.graph(directed)
            slots = api.build_slots(g, 3)
            caches = [api.extract_slot(x, size=4, threshold=4, seed=101 + j)
                      for j, x in enumerate(slots)]
            for node in g.vs["original_id"]:
                checks = api.check_deletion(g, node, 3, [101 + j for j in range(4)],
                                            4, 4, slots, caches)
                self.assertTrue(checks["passed"], checks)
                self.assertEqual(len(checks["unchanged_slots"]), 2)
                self.assertTrue(all(x["input_equal"] and x["cache_equal"] and
                                    x["rng_equal"] for x in checks["comparisons"]))

    def test_slot_stream_is_unchanged_by_other_slot_execution(self):
        api = self.api()
        slots = api.build_slots(self.graph(), 0)
        reference = api.extract_slot(slots[3], size=4, threshold=4, seed=103)
        for _ in range(3):
            api.extract_slot(slots[0], size=4, threshold=4, seed=100)
        later = api.extract_slot(slots[3], size=4, threshold=4, seed=103)
        self.assertEqual(reference["cache"], later["cache"])

    def test_reachability_respects_direction_and_includes_start(self):
        api = self.api()
        g = igraph.Graph(n=5, edges=[(0, 1), (1, 2)], directed=True)
        r = api.reachability(g, 3, 2)
        self.assertEqual(r["out_reachable_ge_scs_fraction"], 1 / 5)
        self.assertEqual(r["out_reachable_ge_bes_fraction"], 2 / 5)
        self.assertEqual(r["out_reachable_max"], 3)
        u = g.as_undirected()
        self.assertEqual(api.reachability(u, 3, 2)["out_reachable_ge_scs_fraction"], 3 / 5)

    def test_saved_decision_mapping_is_validated_and_never_redrawn(self):
        api = self.api()
        g = self.graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.csv"
            rows = [dict(prepared_local_id=i, original_id=g.vs[i]["original_id"],
                         personalized_new_id=i // 2 if i % 2 == 0 else "") for i in range(24)]
            api.write_csv(path, rows)
            sub = api.personalized_graph(g, path)
            self.assertEqual(sub.vs["original_id"], g.vs["original_id"][::2])
            self.assertEqual(sub.vcount(), 12)
            rows[0]["original_id"] = -1
            api.write_csv(path, rows)
            with self.assertRaises(ValueError):
                api.personalized_graph(g, path)

    def test_global_random_state_unchanged(self):
        api = self.api()
        import numpy as np
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        api.extract_slot(self.graph(), size=4, threshold=4, seed=3)
        self.assertEqual(python_before, random.getstate())
        after = np.random.get_state()
        self.assertEqual(numpy_before[0], after[0])
        self.assertTrue(np.array_equal(numpy_before[1], after[1]))
        self.assertEqual(numpy_before[2:], after[2:])

    def test_public_config_has_no_private_statistics_and_output_refuses_overwrite(self):
        api = self.api()
        config = api.public_configuration()
        self.assertEqual(config["partition_seeds"], list(range(5)))
        self.assertEqual(config["quota_per_slot"], 4)
        self.assertNotIn("nodes", config)
        self.assertFalse(config["privacy_claim"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                api.run(Path(tmp), Path(tmp), Path(tmp))

    def test_slot_summary_keeps_five_seeds_per_fixed_slot(self):
        api = self.api()
        rows = [dict(dataset="email", arm="original", slot=slot,
                     partition_seed=seed, subgraphs=slot + seed,
                     container_ge4=slot + seed >= 4)
                for seed in range(5) for slot in range(4)]
        summary = api.slot_summaries(rows, ("subgraphs", "container_ge4"))
        self.assertEqual(len(summary), 4)
        self.assertTrue(all(r["runs"] == 5 for r in summary))
        self.assertEqual(summary[0]["subgraphs_mean"], 2)
        self.assertAlmostEqual(summary[0]["container_ge4_mean"], .2)

    def test_induced_edges_and_duplicate_coverage_are_exact(self):
        api = self.api()
        for directed in (True, False):
            graph = self.graph(directed)
            slots = api.build_slots(graph, 0)
            parent = api.stable_edges(graph)
            for j, slot in enumerate(slots):
                expected = {edge for edge in parent if all(j in api.memberships(v, 0) for v in edge)}
                self.assertEqual(set(api.stable_edges(slot)), expected)
            fake = [{"metrics": dict(container_ge4=False, effective_positions=0, zero_padding=4,
                     structurally_nonempty_contribution=False, subgraphs=0, scs_successes=0,
                     bes_successes=0, walk_attempts=0)} for _ in range(4)]
            result = api.configuration_metrics(graph, slots, fake)
            self.assertEqual(result["effective_positions"], 0)
            self.assertEqual(result["zero_padding"], 16)
            self.assertEqual(result["edge_copies"], result["union_edges"] + result["edges_in_two_slots"])

    def test_locality_detects_corrupted_cache_and_profiler_restores_on_failure(self):
        api = self.api()
        import sys
        g = self.graph()
        slots = api.build_slots(g, 0)
        caches = [api.extract_slot(x, size=4, threshold=4, seed=j) for j, x in enumerate(slots)]
        node = g.vs["original_id"][0]
        untouched = next(j for j in range(4) if j not in api.memberships(node, 0))
        caches[untouched]["cache"]["rng_draw_count"] += 1
        check = api.check_deletion(g, node, 0, list(range(4)), 4, 4, slots, caches)
        self.assertFalse(check["passed"])
        previous = sys.getprofile()
        with self.assertRaises(ValueError):
            api.extract_slot(g, size=4, threshold=0, seed=0)
        self.assertIs(sys.getprofile(), previous)


if __name__ == "__main__":
    unittest.main()

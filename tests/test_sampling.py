import random
import unittest

import igraph

from experiments.sampling import project_max_indegree, sample_privim, sample_privim_star


class SamplingTests(unittest.TestCase):
    def setUp(self):
        # A bidirectional ring is connected, deterministic enough for invariant
        # tests, and has enough distinct nodes for fixed-size walks.
        self.graph = igraph.Graph.Ring(20, directed=True, mutual=True)

    def test_projection_bounds_indegree_without_dropping_vertices(self):
        dense = igraph.Graph.Full(12, directed=True, loops=False)
        projected = project_max_indegree(dense, 3, random.Random(4))
        self.assertEqual(projected.vcount(), dense.vcount())
        self.assertLessEqual(max(projected.indegree()), 3)

    def test_algorithm_one_has_fixed_size_and_r_hop_containment(self):
        projected, result = sample_privim(
            self.graph,
            size=3,
            theta=2,
            r=2,
            sample_rate=1.0,
            restart_probability=0.3,
            walk_length=200,
            rng=random.Random(7),
        )
        self.assertTrue(result.node_sets)
        for start, nodes in enumerate(result.node_sets):
            self.assertEqual(len(nodes), 3)
            self.assertIn(nodes[0], nodes)
            allowed = set(projected.neighborhood(nodes[0], order=2, mode="out"))
            self.assertLessEqual(set(nodes), allowed)

    def test_algorithm_three_never_exceeds_frequency_threshold(self):
        result = sample_privim_star(
            self.graph,
            size=4,
            boundary_divisor=2,
            sample_rate=1.0,
            restart_probability=0.3,
            walk_length=200,
            threshold=3,
            decay=1.5,
            include_boundary_stage=True,
            rng=random.Random(11),
        )
        self.assertTrue(result.node_sets)
        self.assertLessEqual(max(result.frequencies), 3)
        self.assertEqual(
            len(result.node_sets), result.stage1_count + result.stage2_count
        )


if __name__ == "__main__":
    unittest.main()

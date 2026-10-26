import random
import unittest

import igraph
import numpy as np

from experiments.sampling import (
    _released_legacy_walk_from_start,
    project_max_indegree,
    sample_privim,
    sample_privim_star,
    sample_released_legacy,
)


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

    def test_released_walk_uses_all_edges_and_omits_start_vertex(self):
        graph = igraph.Graph(3, directed=True)
        graph.add_edge(0, 2)
        nodes = _released_legacy_walk_from_start(
            graph,
            2,
            size=1,
            restart_probability=0.0,
            max_attempts=10,
            threshold=4,
            decay=1.0,
            frequencies=[0, 0, 0],
            python_rng=random.Random(0),
            numpy_rng=np.random.RandomState(0),
        )
        self.assertEqual(nodes, [0])
        self.assertNotIn(2, nodes)

    def test_released_sampler_retries_to_target_and_caps_boundary_count(self):
        graph = igraph.Graph.Full(60, directed=False, loops=False)
        result = sample_released_legacy(
            graph,
            size=4,
            boundary_divisor=2,
            target_count=16,
            restart_probability=0.3,
            threshold=10,
            decay=1.0,
            include_boundary_stage=True,
            python_rng=random.Random(3),
            numpy_rng=np.random.RandomState(3),
        )
        self.assertEqual(result.stage1_count, 16)
        self.assertEqual(result.stage2_count, 4)
        self.assertEqual(len(result.node_sets), 20)
        self.assertLessEqual(max(result.frequencies), 10)


if __name__ == "__main__":
    unittest.main()

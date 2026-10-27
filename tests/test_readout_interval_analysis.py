"""Exact arithmetic checks for analysis bounds, not model privacy tests."""

from fractions import Fraction
import itertools
import random
import unittest

import numpy as np
import torch
import igraph

from analysis.pdp_head_interval_bounds import (
    candidate_pool, core_coverage_relaxation, minimum_difference_lower, score_error_upper,
)


class ReadoutIntervalTests(unittest.TestCase):
    def test_lower_envelope_contains_exact_rational_minimum(self):
        rng = random.Random(47)
        lower, upper = float(np.float32(.0001)), float(np.float32(.03))
        for dimension in [2, 32]:
            for _ in range(50):
                first = np.array([rng.uniform(-1e5, 1e5) for _ in range(dimension)], dtype=np.float32).astype(np.float64)
                second = np.array([rng.uniform(-1e5, 1e5) for _ in range(dimension)], dtype=np.float32).astype(np.float64)
                exact = Fraction(0)
                for a, b in zip(first, second):
                    difference = Fraction(float(a))-Fraction(float(b))
                    exact += difference*Fraction(lower if difference >= 0 else upper)
                computed = minimum_difference_lower(first, second[None, :], lower, upper)[0]
                self.assertLessEqual(Fraction(float(computed)), exact)

    def test_fp32_linear_error_is_inside_outward_envelope(self):
        rng = random.Random(13)
        upper = float(np.float32(.03))
        for _ in range(80):
            features = np.array([rng.uniform(0, 1e4) for _ in range(32)], dtype=np.float32)
            weights = np.array([rng.uniform(.0001, upper) for _ in range(32)], dtype=np.float32)
            bias = np.float32(rng.uniform(-8, -1))
            observed = float(torch.nn.functional.linear(torch.from_numpy(features), torch.from_numpy(weights)[None, :],
                                                       torch.tensor([bias]))[0])
            exact = sum((Fraction(float(a))*Fraction(float(b)) for a, b in zip(features, weights)), Fraction(float(bias)))
            error = abs(Fraction(observed)-exact)
            bound = score_error_upper(features.astype(np.float64)[None, :], upper, 8)[0]
            self.assertLessEqual(error, Fraction(float(bound)))

    def test_candidate_pool_contains_all_corner_topk_and_excludes_dominated_node(self):
        features = np.array([[1, 1], [2, 0], [0, 0]], dtype=np.float32)
        lower, upper = float(np.float32(.0001)), float(np.float32(.03))
        pool, _ = candidate_pool(features, [0], lower, upper, 8)
        self.assertEqual(pool, {0, 1})
        for weights in itertools.product([lower, upper], repeat=2):
            scores = torch.nn.functional.linear(torch.from_numpy(features), torch.tensor([weights], dtype=torch.float32),
                                                 torch.tensor([-3.0])).reshape(-1)
            selected = int(torch.argsort(scores, descending=True, stable=True)[0])
            self.assertIn(selected, pool)

    def test_integer_core_relaxation_matches_brute_set_union(self):
        graph = igraph.Graph(n=8, edges=[(0, 3), (1, 4), (2, 5), (2, 6)], directed=True)
        pool, core, k = {0, 1, 2, 7}, {0}, 2
        observed, combinations, exact = core_coverage_relaxation(graph, pool, core, k)
        expected = max(len(set(selection) | {u for v in selection for u in graph.successors(v)})
                       for selection in itertools.combinations(pool, k) if core <= set(selection))
        self.assertEqual(observed, expected)
        self.assertEqual(combinations, 3)
        self.assertTrue(exact)
        union_bound, _, exact = core_coverage_relaxation(graph, pool, core, k, maximum_combinations=1)
        self.assertGreaterEqual(union_bound, expected)
        self.assertFalse(exact)
        large = igraph.Graph(n=130, edges=[(0, 129)], directed=True)
        self.assertEqual(core_coverage_relaxation(large, {0}, {0}, 1)[0], 2)


if __name__ == "__main__":
    unittest.main()

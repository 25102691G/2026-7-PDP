import unittest

import igraph
import numpy as np

from experiments.audit_oracle import spearman, topk_by_singleton_influence


class OracleAuditTests(unittest.TestCase):
    def test_singleton_ranking_breaks_degree_ties_by_node_id(self):
        graph = igraph.Graph(n=5, edges=[(0, 3), (1, 3), (2, 4)], directed=True)
        self.assertEqual(topk_by_singleton_influence(graph, 3), [0, 1, 2])

    def test_spearman_uses_average_tie_ranks(self):
        left = np.asarray([1.0, 2.0, 2.0, 4.0])
        right = np.asarray([10.0, 20.0, 20.0, 40.0])
        self.assertAlmostEqual(spearman(left, right), 1.0)


if __name__ == "__main__":
    unittest.main()

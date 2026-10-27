"""Node-PDP sampling invariants and checks of the proved event inequalities."""

from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction
import itertools
import math
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DGLBACKEND", "pytorch")
import igraph
import torch

from pdp.accounting import RangePrivacy
from pdp.personalization import (
    HashBudgetPolicy, NodeBudget, amplified_bounds, retention_probability,
    sample_nodes_once, train_pdp,
)
from pdp.training import TrainingConfig, train_b0


class PDPSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def graph(self):
        graph = igraph.Graph(n=4, edges=[(0, 1), (1, 2)], directed=True)
        graph.vs["stable_id"] = [10, 20, 30, 40]
        graph.vs["private_value"] = ["a", "b", "c", "isolated"]
        return graph

    def test_probabilities_are_conservative_and_bounds_meet_epsilon_delta(self):
        for epsilon_0 in ["1", "2", "4"]:
            for epsilon in ["0", "0.0000000000000000000001", "1", "2", "4", "1000"]:
                for delta in ["0", "0.000000001", "0.000001"]:
                    budget = NodeBudget(epsilon, delta)
                    probability = retention_probability(budget, epsilon_0, "0.000001")
                    upper_epsilon, upper_delta = amplified_bounds(
                        probability, epsilon_0, "0.000001",
                    )
                    self.assertLessEqual(upper_epsilon, Decimal(epsilon))
                    self.assertLessEqual(upper_delta, Fraction(delta))
                    if Decimal(epsilon) < Decimal(epsilon_0):
                        with localcontext() as context:
                            context.prec = 150
                            exact_ratio = (
                                (Decimal(epsilon).exp()-1)/(Decimal(epsilon_0).exp()-1)
                            )
                            actual = Decimal(probability.numerator)/probability.denominator
                        self.assertLessEqual(actual, exact_ratio)

    def test_known_retention_rates_and_no_need_to_choose_maximum_base_budget(self):
        expected = {
            "1": [1, 1, 1],
            "2": [0.2689414213699951, 1, 1],
            "4": [0.03205860328008499, 0.11920292202211755, 1],
        }
        for base, values in expected.items():
            probabilities = [
                retention_probability(NodeBudget(value, "0.000001"), base, "0.000001")
                for value in ["1", "2", "4"]
            ]
            for probability, value in zip(probabilities, values):
                self.assertAlmostEqual(float(probability), value, places=14)

    def test_deletion_preserves_other_budgets_ids_attributes_and_incident_edges(self):
        graph = self.graph()
        policy = HashBudgetPolicy()
        other = graph.copy()
        other.delete_vertices([0])
        self.assertEqual(other.vs["stable_id"], [20, 30, 40])
        before = {identifier: policy.group_for(identifier)
                  for identifier in graph.vs["stable_id"]}
        after = {identifier: policy.group_for(identifier)
                 for identifier in other.vs["stable_id"]}
        self.assertEqual(after, {key: before[key] for key in after})
        # Coupling all remaining nodes as retained creates exactly node
        # adjacency; source attributes are restricted along with vertices.
        with patch("pdp.personalization.secrets.randbelow", return_value=0):
            first = sample_nodes_once(graph, policy, "4", "0.000001")
            second = sample_nodes_once(other, policy, "4", "0.000001")
        first.delete_vertices(first.vs.find(stable_id=10).index)
        self.assertEqual(first.vs["stable_id"], second.vs["stable_id"])
        self.assertEqual(first.vs["private_value"], second.vs["private_value"])
        self.assertEqual(first.get_edgelist(), second.get_edgelist())

    def test_zero_budget_removes_all_information_and_preserves_empty_graph_validity(self):
        policy = HashBudgetPolicy(groups=(NodeBudget("0", "0"),))
        retained = sample_nodes_once(self.graph(), policy, "2", "0.000001")
        self.assertEqual(retained.vcount(), 0)
        self.assertEqual(retained.ecount(), 0)
        second = sample_nodes_once(retained, policy, "2", "0.000001")
        self.assertEqual(second.vcount(), 0)

    def test_full_participation_keeps_isolates_and_uniform_case_equals_base(self):
        graph = self.graph()
        policy = HashBudgetPolicy(groups=(NodeBudget("2", "0.000001"),))
        retained = sample_nodes_once(graph, policy, "2", "0.000001")
        self.assertEqual(retained.vcount(), 4)
        self.assertEqual(retained.vs["private_value"][-1], "isolated")
        config = TrainingConfig(
            privacy=RangePrivacy(epsilon="2", steps=2, batch_size=2, grid_bits=8),
            hidden_dim=4, subgraph_size=2, frequency_threshold=2,
        )
        with patch("pdp.training.DiscreteGaussian.sample",
                   side_effect=lambda dimension: [1]*dimension):
            base = train_b0(graph, config)
            personalized = train_pdp(graph, config, policy)
        for key in base.parameters:
            self.assertTrue(torch.equal(base.parameters[key], personalized.parameters[key]))
        self.assertEqual(personalized.certificate["personalization"]["outer_sampling_count"], 1)

    def test_outer_sampler_called_once_not_once_per_optimizer_update(self):
        config = TrainingConfig(
            privacy=RangePrivacy(steps=3, batch_size=2, grid_bits=8),
            hidden_dim=4, subgraph_size=2,
        )
        with patch("pdp.personalization.sample_nodes_once",
                   wraps=sample_nodes_once) as outer:
            result = train_pdp(self.graph(), config, HashBudgetPolicy())
        self.assertEqual(outer.call_count, 1)
        self.assertNotIn("outer_retained_nodes", result.certificate)
        self.assertNotIn("stable_id", str(result.certificate))

    def test_enumerated_randomized_response_events_satisfy_both_directions(self):
        # All events of a two-element output universe, for the tight pure-DP
        # membership randomized-response mechanism used in the proof.
        for base_epsilon in [1, 2, 4]:
            E = math.exp(base_epsilon)
            P = [1/(E+1), E/(E+1)]
            Q = list(reversed(P))
            for p in [0, 0.03, 0.27, 1]:
                R = [(1-p)*a+p*b for a, b in zip(P, Q)]
                multiplier = 1+p*(E-1)
                for event in itertools.product([False, True], repeat=2):
                    prob_p = sum(P[i] for i in range(2) if event[i])
                    prob_r = sum(R[i] for i in range(2) if event[i])
                    self.assertLessEqual(prob_r, multiplier*prob_p+1e-14)
                    self.assertLessEqual(prob_p, multiplier*prob_r+1e-14)


if __name__ == "__main__":
    unittest.main()

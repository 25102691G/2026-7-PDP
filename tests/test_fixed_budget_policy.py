"""External preferences stay fixed on the whole domain, including absent IDs."""

from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DGLBACKEND", "pytorch")
import igraph
import torch

from pdp.accounting import RangePrivacy
from pdp.personalization import NodeBudget, sample_nodes_once, train_pdp
from pdp.policies import FixedBudgetPolicy
from pdp.stable_slots import StableSlots, slot_accounting
from pdp.training import TrainingConfig


class FixedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_public_default_and_original_dictionary_mutation(self):
        overrides = {10: NodeBudget("0", "0"), 20: NodeBudget("4", "0.000001")}
        policy = FixedBudgetPolicy(NodeBudget("1", "0.000001"), overrides)
        original = {identifier: policy.groups[policy.group_for(identifier)] for identifier in [10, 20, 30, 99999]}
        overrides[20] = NodeBudget("0", "0")
        overrides[30] = NodeBudget("9", "0.00001")
        self.assertEqual(policy.groups[policy.group_for(20)], original[20])
        self.assertEqual(policy.groups[policy.group_for(30)], original[30])
        self.assertEqual(policy.group_for(99999), 0)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            policy._group_by_id = {}
        with self.assertRaises(TypeError):
            policy._group_by_id[20] = 0

    def test_both_epsilon_and_delta_constraints_and_public_certificate(self):
        policy = FixedBudgetPolicy(NodeBudget("1", "0.000001"), {
            10: NodeBudget("2", "0.000000001"), 20: NodeBudget("4", "0.000001"),
        })
        certificate = policy.certificate("2", "0.000001")
        self.assertEqual(certificate["policy"], "public_fixed_budget_mapping_with_default")
        self.assertNotIn("overrides", certificate)
        self.assertNotIn("members", certificate)
        for target, bound in zip(policy.groups, certificate["groups"]):
            self.assertLessEqual(Decimal(bound["epsilon_upper"]), Decimal(target.epsilon))
            self.assertLessEqual(Fraction(bound["delta_upper_numerator"], bound["delta_upper_denominator"]),
                                 Fraction(target.delta))

    def test_stable_policy_identity_does_not_depend_on_dictionary_order(self):
        default = NodeBudget("1", "0.000001")
        first = FixedBudgetPolicy(default, {20: NodeBudget("4", "0.000001"), 10: NodeBudget("2", "0.000001")})
        second = FixedBudgetPolicy(default, {10: NodeBudget("2", "0.000001"), 20: NodeBudget("4", "0.000001")})
        self.assertEqual(first.public_description(), second.public_description())
        self.assertEqual(first.groups, second.groups)

    def test_zero_budget_deletion_has_no_effect_in_actual_outer_wrapper(self):
        graph = igraph.Graph.Full(5, directed=True, loops=False)
        graph.vs["stable_id"] = [10, 20, 30, 40, 50]
        policy = FixedBudgetPolicy(NodeBudget("1", "0.000001"), {10: NodeBudget("0", "0")})
        other = graph.copy()
        other.delete_vertices(0)
        config = TrainingConfig(privacy=RangePrivacy(epsilon="2", steps=1, batch_size=2, grid_bits=8),
                                hidden_dim=4, subgraph_size=2, walk_length=10)
        with patch("pdp.personalization.secrets.randbelow", return_value=0):
            first_sample = sample_nodes_once(graph, policy, "2", "0.000001")
            second_sample = sample_nodes_once(other, policy, "2", "0.000001")
            self.assertEqual(first_sample.vs["stable_id"], second_sample.vs["stable_id"])
            with patch("pdp.training.DiscreteGaussian.sample", side_effect=lambda d: [1]*d):
                first = train_pdp(graph, config, policy)
                second = train_pdp(other, config, policy)
        self.assertTrue(all(torch.equal(first.parameters[k], second.parameters[k]) for k in first.parameters))

    def test_absent_strict_id_still_affects_public_noise_calibration(self):
        policy = FixedBudgetPolicy(NodeBudget("1", "0.000001"), {999999: NodeBudget("0.1", "0.000001")})
        config = TrainingConfig(privacy=RangePrivacy(batch_size=2, steps=2), hidden_dim=4)
        slots = StableSlots(rounds=1, bins_per_round=2, appearance_limits=(1, 1))
        sigma, certificate = slot_accounting(config, policy, slots)
        uniform = FixedBudgetPolicy(NodeBudget("1", "0.000001"), {})
        ordinary, _ = slot_accounting(config, uniform, StableSlots(rounds=1, bins_per_round=2, appearance_limits=(1,)))
        self.assertGreater(sigma, ordinary)
        self.assertEqual(certificate["policy"], "public_fixed_budget_mapping_with_default")


if __name__ == "__main__":
    unittest.main()

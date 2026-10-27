"""Stable-slot invariants required by the explicit Node-PDP proof."""

from dataclasses import replace
from decimal import Decimal
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DGLBACKEND", "pytorch")
import igraph
import torch

from pdp.accounting import RangePrivacy
from pdp.personalization import HashBudgetPolicy, NodeBudget
from pdp.stable_slots import (
    StableSlots, _canonical_slot_graph, _prepare_slot_samples, _slot_contribution,
    slot_accounting, slot_members, train_slots,
)
from pdp.training import TrainingConfig, initialize_model


class StableSlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def graph(self, count=18):
        graph = igraph.Graph.Full(count, directed=True, loops=False)
        graph.vs["stable_id"] = list(range(100, 100+count))
        graph.vs["private_attribute"] = list(range(count))
        return graph

    def config(self):
        return TrainingConfig(
            privacy=RangePrivacy(steps=2, batch_size=6, grid_bits=8),
            hidden_dim=4, subgraph_size=2, frequency_threshold=2, walk_length=10,
        )

    def slots(self):
        return StableSlots(bins_per_round=2, rounds=3, appearance_limits=(1, 2, 3))

    def test_policy_and_slot_limits_do_not_follow_mutated_caller_lists(self):
        groups = [NodeBudget("1", "0.000001")]
        policy = HashBudgetPolicy(groups=groups)
        limits = [1]
        slots = StableSlots(appearance_limits=limits)
        groups.append(NodeBudget("2", "0.000001"))
        limits[0] = 4
        self.assertEqual(len(policy.groups), 1)
        self.assertEqual(slots.appearance_limits, (1,))

    def test_deletion_changes_exactly_assigned_slots_and_restricts_attributes_edges(self):
        graph, policy, slots = self.graph(), HashBudgetPolicy(), self.slots()
        original = slot_members(graph, policy, slots)
        original_ids = [set(graph.vs[x]["stable_id"]) for x in original]
        for index, identifier in enumerate(graph.vs["stable_id"]):
            other = graph.copy()
            other.delete_vertices(index)
            after = slot_members(other, policy, slots)
            after_ids = [set(other.vs[x]["stable_id"]) for x in after]
            affected = {j for j, members in enumerate(original_ids) if identifier in members}
            self.assertEqual(len(affected), slots.appearance_limits[policy.group_for(identifier)])
            self.assertEqual({j for j, (a, b) in enumerate(zip(original_ids, after_ids)) if a != b}, affected)
            for j, (before, retained) in enumerate(zip(original, after)):
                first, second = graph.induced_subgraph(before), other.induced_subgraph(retained)
                if j in affected:
                    first.delete_vertices(first.vs.find(stable_id=identifier).index)
                self.assertEqual(first.vs["stable_id"], second.vs["stable_id"])
                self.assertEqual(first.vs["private_attribute"], second.vs["private_attribute"])
                self.assertEqual(first.get_edgelist(), second.get_edgelist())

    def test_real_gradients_and_rng_of_unaffected_slots_are_identical(self):
        graph, policy, slots, config = self.graph(), HashBudgetPolicy(), self.slots(), self.config()
        victim = next(v.index for v in graph.vs if policy.group_for(v["stable_id"]) == 0)
        identifier = graph.vs[victim]["stable_id"]
        other = graph.copy()
        other.delete_vertices(victim)
        members = slot_members(graph, policy, slots)
        affected = {j for j, indices in enumerate(members) if victim in indices}
        containers = _prepare_slot_samples(graph, members, config, {})
        other_containers = _prepare_slot_samples(other, slot_members(other, policy, slots), config, {})
        model = initialize_model(config)
        dimension = sum(p.numel() for p in model.parameters())
        totals = [[0]*dimension, [0]*dimension]
        nonzero = 0
        for index, (first, second) in enumerate(zip(containers, other_containers)):
            a = _slot_contribution(model, first, config, slots, index, 1, dimension)
            b = _slot_contribution(model, second, config, slots, index, 1, dimension)
            nonzero += int(any(a))
            self.assertLessEqual(sum(x*x for x in a), config.privacy.radius**2)
            self.assertLessEqual(sum(x*x for x in b), config.privacy.radius**2)
            if index not in affected:
                self.assertEqual(a, b)
            totals[0] = [x+y for x, y in zip(totals[0], a)]
            totals[1] = [x+y for x, y in zip(totals[1], b)]
        self.assertGreater(nonzero, 0)
        sensitivity = 2*slots.appearance_limits[policy.group_for(identifier)]*config.privacy.radius
        self.assertLessEqual(sum((x-y)**2 for x, y in zip(*totals)), sensitivity**2)

    def test_local_renumbering_and_edge_order_cannot_change_slot_queries(self):
        graph, policy, slots, config = self.graph(), HashBudgetPolicy(), self.slots(), self.config()
        order = list(reversed(range(graph.vcount())))
        mapping = {old: new for new, old in enumerate(order)}
        other = igraph.Graph(n=graph.vcount(), directed=True, edges=[
            (mapping[a], mapping[b]) for a, b in reversed(graph.get_edgelist())
        ])
        for attribute in graph.vs.attributes():
            other.vs[attribute] = [graph.vs[i][attribute] for i in order]
        containers = _prepare_slot_samples(graph, slot_members(graph, policy, slots), config, {})
        other_containers = _prepare_slot_samples(other, slot_members(other, policy, slots), config, {})
        model = initialize_model(config)
        dimension = sum(p.numel() for p in model.parameters())
        for index, (first, second) in enumerate(zip(containers, other_containers)):
            self.assertEqual(
                _slot_contribution(model, first, config, slots, index, 0, dimension),
                _slot_contribution(model, second, config, slots, index, 0, dimension),
            )

    def test_one_failed_slot_does_not_clear_or_reseed_other_slots(self):
        graph, policy, slots, config = self.graph(), HashBudgetPolicy(), self.slots(), self.config()
        members = slot_members(graph, policy, slots)
        seen_seeds = []
        def fake_build(local, local_config):
            seen_seeds.append(local_config.sampling_seed)
            if len(seen_seeds) == 2:
                raise RuntimeError("private local error")
            return [tuple(local.vs["stable_id"])]
        diagnostic = {}
        with patch("pdp.stable_slots.build_samples", side_effect=fake_build):
            containers = _prepare_slot_samples(graph, members, config, diagnostic)
        self.assertEqual(diagnostic["preprocessing_failed_slots"], [1])
        self.assertEqual(containers[1], [])
        self.assertTrue(all(len(x) == 1 for j, x in enumerate(containers) if j != 1))
        self.assertEqual(len(set(seen_seeds)), slots.count)

    def test_public_group_noise_and_certificates_cover_groups_absent_from_input(self):
        config, policy, slots = TrainingConfig(), HashBudgetPolicy(), StableSlots()
        sigma, certificate = slot_accounting(config, policy, slots)
        self.assertLess(sigma, config.privacy.calibrate()/10)
        for group, bound in zip(policy.groups, certificate["groups"]):
            self.assertLessEqual(Decimal(bound["epsilon_upper"]), Decimal(group.epsilon))
            self.assertLessEqual(Decimal(bound["delta_upper"]), Decimal(group.delta))
        low, _ = slot_accounting(config, policy, replace(slots, appearance_limits=(1, 1, 1),
                                                        uniform_epsilon="1", uniform_delta="0.000001"))
        full, _ = slot_accounting(config, policy, replace(slots, appearance_limits=(4, 4, 4),
                                                         uniform_epsilon="1", uniform_delta="0.000001"))
        self.assertLess(low, sigma)
        self.assertGreater(full, sigma)
        for forbidden in ["slot_node_counts", "subgraph_count", "train_nodes", "sampling_mask"]:
            self.assertNotIn(forbidden, certificate)

    def test_fixed_T_on_bare_empty_and_nonempty_graphs_with_identical_public_certificate(self):
        config, slots, policy = self.config(), self.slots(), HashBudgetPolicy()
        previous = None
        for graph in [igraph.Graph(n=0, directed=True), self.graph()]:
            diagnostic = {}
            with patch("pdp.stable_slots.DiscreteGaussian.sample",
                       side_effect=lambda dimension: [1]*dimension) as noise:
                release = train_slots(graph, config, policy, slots, trusted_diagnostics=diagnostic)
            self.assertEqual(noise.call_count, config.privacy.steps)
            self.assertEqual(diagnostic["completed_steps"], config.privacy.steps)
            self.assertEqual(diagnostic["invalid_slot_queries"], 0)
            self.assertTrue(all(bool(torch.isfinite(p).all()) for p in release.parameters.values()))
            if previous is not None:
                self.assertEqual(previous, release.certificate)
            previous = release.certificate

    def test_zero_budget_nodes_have_no_route_and_exactly_no_effect_with_coupled_noise(self):
        policy = HashBudgetPolicy(groups=(NodeBudget("0", "0"), NodeBudget("2", "0.000001")))
        slots = StableSlots(bins_per_round=2, rounds=3, appearance_limits=(0, 3))
        graph = self.graph()
        victim = next(v.index for v in graph.vs if policy.group_for(v["stable_id"]) == 0)
        other = graph.copy()
        other.delete_vertices(victim)
        with patch("pdp.stable_slots.DiscreteGaussian.sample", side_effect=lambda d: [1]*d):
            first = train_slots(graph, self.config(), policy, slots)
            second = train_slots(other, self.config(), policy, slots)
        self.assertEqual(first.certificate["groups"][0]["epsilon_upper"], "0")
        self.assertEqual(first.certificate["groups"][0]["delta_upper"], "0")
        self.assertTrue(all(torch.equal(first.parameters[k], second.parameters[k]) for k in first.parameters))

    def test_no_noise_control_is_never_labelled_private(self):
        with patch("pdp.stable_slots.DiscreteGaussian.sample") as noise:
            release = train_slots(self.graph(), self.config(), HashBudgetPolicy(),
                                  self.slots(), noise_enabled=False)
        noise.assert_not_called()
        self.assertEqual(release.certificate["privacy_guarantee"], "none")
        self.assertNotIn("groups", release.certificate)

    def test_invalid_public_layout_or_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            StableSlots(appearance_limits=(5,))
        with self.assertRaises(ValueError):
            slot_accounting(self.config(), HashBudgetPolicy(), StableSlots())
        with self.assertRaises(ValueError):
            slot_accounting(self.config(), HashBudgetPolicy(), replace(self.slots(),
                uniform_epsilon="2", uniform_delta="0.000001"))
        with self.assertRaises(ValueError):
            slot_accounting(self.config(), HashBudgetPolicy(groups=(NodeBudget("0", "0"),)),
                            replace(self.slots(), appearance_limits=(1,)))

    def test_sampling_quantities_are_explicit_without_publishing_private_rates(self):
        config, slots, policy = self.config(), self.slots(), HashBudgetPolicy()
        diagnostic = {}
        with patch("pdp.stable_slots.DiscreteGaussian.sample", side_effect=lambda d: [1]*d):
            release = train_slots(self.graph(), config, policy, slots, trusted_diagnostics=diagnostic)
        metadata = release.certificate["sampling_accounting"]
        self.assertEqual(metadata["n"], config.subgraph_size)
        self.assertEqual(metadata["r"], 3)
        self.assertIsNone(metadata["theta"])
        self.assertEqual(metadata["M_internal"], config.frequency_threshold)
        self.assertIn("slot_nodes", metadata["q_rule"])
        self.assertEqual(diagnostic["sampling_rate_by_slot"], [
            min(1, config.expected_starts/max(1, size)) for size in diagnostic["slot_node_counts"]
        ])
        self.assertNotIn("sampling_rate_by_slot", release.certificate)
        self.assertNotIn("slot_node_counts", release.certificate)

    def test_undirected_edge_orientation_is_canonical_before_library_simplification(self):
        graph = igraph.Graph(n=5, edges=[(4, 0), (3, 2), (1, 4), (1, 3)])
        graph.vs["stable_id"] = [50, 40, 30, 20, 10]
        order = list(reversed(range(5)))
        mapping = {old: new for new, old in enumerate(order)}
        other = igraph.Graph(n=5, edges=[
            (mapping[a], mapping[b]) for a, b in reversed(graph.get_edgelist())
        ])
        other.vs["stable_id"] = [graph.vs[i]["stable_id"] for i in order]
        first = _canonical_slot_graph(graph, list(range(5)))
        second = _canonical_slot_graph(other, list(range(5)))
        self.assertEqual(first.vs["stable_id"], second.vs["stable_id"])
        self.assertEqual(first.get_edgelist(), second.get_edgelist())


if __name__ == "__main__":
    unittest.main()

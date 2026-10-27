"""Readout subspace, public initialization, and post-noise constraints."""

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("DGLBACKEND", "pytorch")
import igraph
import torch

from pdp.accounting import RangePrivacy
from pdp.personalization import HashBudgetPolicy, NodeBudget
from pdp.positive_head import (
    PositiveHeadSettings, apply_head_update, head_contribution,
    initialize_positive_head, positive_initialization_release,
)
from pdp.stable_slots import StableSlots, slot_accounting, train_slots
from pdp.training import TrainingConfig, build_samples, initialize_model


class PositiveHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return TrainingConfig(
            privacy=RangePrivacy(steps=3, batch_size=6, grid_bits=16),
            learning_rate=0.0001, hidden_dim=4, subgraph_size=2,
            frequency_threshold=2, walk_length=10,
        )

    def slots(self):
        return StableSlots(bins_per_round=2, rounds=3, appearance_limits=(1, 2, 3),
                           trainable_mode="positive_head")

    def graph(self):
        graph = igraph.Graph.Full(18, directed=True, loops=False)
        graph.vs["stable_id"] = list(range(18))
        return graph

    def test_public_encoder_and_exact_trainable_subspace(self):
        config, settings = self.config(), PositiveHeadSettings()
        first = initialize_positive_head(config, settings)
        second = initialize_positive_head(config, settings)
        self.assertEqual([name for name, p in first.named_parameters() if p.requires_grad],
                         ["grat.grat3.fc.weight", "grat.grat3.fc.bias"])
        self.assertEqual(sum(p.numel() for p in first.parameters() if p.requires_grad), 5)
        self.assertTrue(all(torch.equal(value, second.state_dict()[name])
                            for name, value in first.state_dict().items()))
        for layer in [first.grat.grat1, first.grat.grat2]:
            self.assertTrue(bool((layer.fc.weight >= 0).all()))
            self.assertTrue(bool((layer.fc.bias > 0).all()))
        for layer in [first.grat.grat1, first.grat.grat2, first.grat.grat3]:
            self.assertTrue(bool((layer.attn_fc.weight == 0).all()))
            self.assertTrue(bool((layer.attn_fc.bias == 1).all()))

    def test_actual_equation_five_gradient_is_nonzero_and_only_for_head(self):
        config = self.config()
        model = initialize_positive_head(config, PositiveHeadSettings())
        vector = head_contribution(model, build_samples(self.graph(), config)[0], config)
        self.assertEqual(len(vector), config.hidden_dim+1)
        self.assertTrue(any(vector))
        self.assertLessEqual(sum(x*x for x in vector), config.privacy.radius**2)

    def test_training_changes_head_preserves_every_public_frozen_parameter(self):
        config, slots = self.config(), self.slots()
        model = initialize_positive_head(config, slots.head_settings)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        diagnostic = {}
        with patch("pdp.stable_slots._contribution", side_effect=AssertionError("frozen autograd misuse")):
            with patch("pdp.stable_slots.DiscreteGaussian.sample",
                       side_effect=lambda d: [1]*d) as noise:
                result = train_slots(self.graph(), config, HashBudgetPolicy(), slots,
                                     trusted_diagnostics=diagnostic)
        self.assertEqual(noise.call_count, config.privacy.steps)
        self.assertTrue(all(call.args == (5,) for call in noise.call_args_list))
        self.assertEqual(result.certificate["private_parameter_count"], 5)
        self.assertGreater(result.certificate["parameter_count"], 5)
        self.assertGreater(diagnostic["nonzero_slot_queries"], 0)
        self.assertEqual(diagnostic["invalid_slot_queries"], 0)
        self.assertTrue(any(not torch.equal(before[name], result.parameters[name]) for name in trainable))
        self.assertTrue(all(torch.equal(value, result.parameters[name])
                            for name, value in before.items() if name not in trainable))

    def test_extreme_integer_noise_is_postprocessed_and_cannot_modify_encoder(self):
        config, settings = self.config(), PositiveHeadSettings()
        model = initialize_positive_head(config, settings)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        for sign in [-1, 1]:
            apply_head_update(model, [sign*10**100]*5, config, settings)
            head = model.grat.grat3.fc
            self.assertTrue(bool((head.weight >= settings.weight_lower).all()))
            self.assertTrue(bool((head.weight <= settings.weight_upper).all()))
            self.assertTrue(bool((head.bias >= settings.bias_lower).all()))
            self.assertTrue(bool((head.bias <= settings.bias_upper).all()))
            self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()))

    def test_public_initialization_matches_same_encoder_and_reloads_original_architecture(self):
        config, settings = self.config(), PositiveHeadSettings()
        release = positive_initialization_release(config, settings)
        model = initialize_positive_head(config, settings)
        reconstructed = initialize_model(config)
        reconstructed.load_state_dict(release.parameters)
        self.assertEqual(release.certificate["privacy_guarantee"], "data_independent")
        self.assertEqual(release.certificate["optimizer_steps"], 0)
        sample = build_samples(self.graph(), config)[0][0]
        with torch.no_grad():
            self.assertTrue(torch.equal(model.logits(sample, sample.ndata["feat"]),
                                        reconstructed.logits(sample, sample.ndata["feat"])))

    def test_same_required_noise_and_heterogeneous_deltas(self):
        config, slots = self.config(), self.slots()
        policy = HashBudgetPolicy(groups=(NodeBudget("1", "0.000000001"),
                                         NodeBudget("2", "0.000001"), NodeBudget("4", "0.0000001")))
        sigma, certificate = slot_accounting(config, policy, slots)
        full_sigma, _ = slot_accounting(config, policy, replace(slots, trainable_mode="full"))
        self.assertEqual(sigma, full_sigma)
        self.assertEqual([g["delta_upper"] for g in certificate["groups"]],
                         [g.delta for g in policy.groups])

    def test_matched_zero_noise_control_is_explicitly_nonprivate(self):
        with patch("pdp.stable_slots.DiscreteGaussian.sample") as noise:
            release = train_slots(self.graph(), self.config(), HashBudgetPolicy(),
                                  self.slots(), noise_enabled=False)
        noise.assert_not_called()
        self.assertEqual(release.certificate["privacy_guarantee"], "none")
        self.assertNotIn("groups", release.certificate)


if __name__ == "__main__":
    unittest.main()

"""Joint per-subgraph clipping checks, not DP/PDP certification."""

import importlib.util
import unittest
from unittest.mock import patch


class NodeSamplingClippingTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.node_sampling_clipping"),
                             "independent clipping entry has not been implemented")
        from experiments import node_sampling_clipping
        return node_sampling_clipping

    def test_joint_clipping_across_parameter_groups_and_exact_sgd_update(self):
        api = self.api()
        torch = api.torch
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = api.ObservedPrivateSGD(
            [{"params": [first]}, {"params": [second]}],
            l2_norm_clip=3.0, minibatch_size=2, lr=0.1,
        )
        optimizer.zero_accum_grad()
        first.grad = torch.tensor([3.0])
        second.grad = torch.tensor([4.0])
        optimizer.microbatch_step()
        self.assertAlmostEqual(optimizer.current_samples[0]["pre_norm"], 5.0)
        self.assertAlmostEqual(optimizer.current_samples[0]["post_norm"], 3.0, places=5)
        first.grad = torch.tensor([0.0])
        second.grad = torch.tensor([-2.0])
        optimizer.microbatch_step()
        with patch.object(torch, "randn", side_effect=AssertionError("noise forbidden")), \
             patch.object(torch, "randn_like", side_effect=AssertionError("noise forbidden")):
            optimizer.step()
        coefficient = 3.0 / (5.0 + 1e-6)
        self.assertAlmostEqual(first.item(), 1.0 - 0.1 * (3.0 * coefficient / 2), places=6)
        self.assertAlmostEqual(second.item(), 2.0 - 0.1 * ((4.0 * coefficient - 2.0) / 2), places=6)
        self.assertEqual(optimizer.records[0]["clip_fraction"], 0.5)
        self.assertTrue(optimizer.records[0]["update_verified"])
        self.assertTrue(optimizer.records[0]["joint_bound_ok"])
        self.assertEqual(optimizer.records[0]["noise_norm"], 0.0)

    def test_zero_gradients_are_finite_and_not_marked_clipped(self):
        api = self.api()
        parameter = api.torch.nn.Parameter(api.torch.zeros(2))
        optimizer = api.ObservedPrivateSGD([parameter], l2_norm_clip=3.0, minibatch_size=1, lr=0.005)
        optimizer.zero_accum_grad()
        parameter.grad = api.torch.zeros(2)
        optimizer.microbatch_step()
        optimizer.step()
        self.assertEqual(optimizer.records[0]["mean_gradient_norm"], 0.0)
        self.assertEqual(optimizer.records[0]["clip_fraction"], 0.0)

    def test_incomplete_batch_or_nonfinite_gradients_are_rejected(self):
        api = self.api()
        parameter = api.torch.nn.Parameter(api.torch.zeros(1))
        optimizer = api.ObservedPrivateSGD([parameter], l2_norm_clip=3.0, minibatch_size=2, lr=0.005)
        optimizer.zero_accum_grad()
        parameter.grad = api.torch.tensor([float("nan")])
        with self.assertRaises(ValueError):
            optimizer.microbatch_step()
        parameter.grad = api.torch.ones(1)
        optimizer.microbatch_step()
        with self.assertRaises(ValueError):
            optimizer.step()

    def test_initial_ties_are_measured_on_the_fixed_test_graph(self):
        api = self.api()
        rt = api.u.runtime()
        _, graph, directed = rt["loader"].load_prepared_graph("email", "test", api.u.ROOT / "dataset/prepared")
        dg = rt["graph_util"].get_rev_dgl(graph, "ones", 32, directed, False)
        profiles = [api.audit_initialization(api.u.new_model(seed), graph, dg, 432) for seed in (1, 2)]
        for profile in profiles:
            self.assertGreater(profile["unique_logits"], 1)
            self.assertEqual(profile["boundary_tie_count"], 79)
            self.assertEqual(profile["max_tie_count"], 79)
            self.assertFalse(profile["all_logits_equal"])
            self.assertEqual(profile["selected_zero_outdegree_count"], 50)
        self.assertEqual(profiles[0]["selected_local_ids"], profiles[1]["selected_local_ids"])
        self.assertNotEqual(profiles[0]["initial_state_sha256"], profiles[1]["initial_state_sha256"])

    def test_zero_noise_constructor_has_no_noise_configuration(self):
        api = self.api()
        parameter = api.torch.nn.Parameter(api.torch.zeros(1))
        with self.assertRaises(TypeError):
            api.ObservedPrivateSGD([parameter], l2_norm_clip=3, minibatch_size=1, lr=.005, noise_multiplier=1)


if __name__ == "__main__":
    unittest.main()

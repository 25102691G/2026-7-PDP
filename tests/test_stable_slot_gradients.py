"""Synthetic vector tests only; never train or run a noisy optimizer."""
import importlib.util
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


class StableSlotGradientTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.stable_slot_gradients"))
        from experiments import stable_slot_gradients
        return stable_slot_gradients

    def test_joint_clip_matches_original_method_all_2340_coordinates(self):
        api = self.api()
        from experiments.privacy import PrivateSGD
        parameters = [torch.nn.Parameter(torch.zeros(1000)), torch.nn.Parameter(torch.zeros(1340))]
        gradients = [torch.linspace(-2, 3, 1000), torch.linspace(4, -1, 1340)]
        groups = []
        for p, g in zip(parameters, gradients):
            p.grad = g.clone()
            groups.append(dict(params=[p], accum_grads=[torch.zeros_like(p)]))
        carrier = SimpleNamespace(param_groups=groups, last_microbatch_norms=[], l2_norm_clip=3.)
        # Invoke only original microbatch clipping, without constructing an optimizer.
        original_norm = PrivateSGD.microbatch_step(carrier)
        clipped, info = api.clip_joint(gradients, parameters)
        expected = torch.cat([g["accum_grads"][0] for g in groups])
        self.assertTrue(torch.equal(clipped, expected))
        self.assertEqual(clipped.numel(), 2340)
        self.assertEqual(info["legacy_pre_norm"], original_norm)
        self.assertLessEqual(info["post_norm"], 3.00001)

    def test_joint_not_per_parameter_clipping_missing_zero_and_nonfinite(self):
        api = self.api()
        templates = [torch.zeros(1), torch.zeros(1), torch.zeros(2)]
        clipped, info = api.clip_joint([torch.tensor([3.]), torch.tensor([4.]), None], templates)
        self.assertTrue(torch.allclose(clipped, torch.tensor([1.8, 2.4, 0., 0.]), atol=1e-6))
        self.assertEqual(info["missing_coordinates"], 2)
        self.assertEqual(info["missing_parameters"], 1)
        z, zi = api.clip_joint([torch.zeros(1), None, torch.zeros(2)], templates)
        self.assertEqual(api.norm(z), 0.)
        self.assertEqual(zi["coefficient"], 1.)
        with self.assertRaises(ValueError):
            api.clip_joint([torch.tensor([float('nan')]), torch.ones(1), None], templates)

    def test_slot_buffers_are_independent_and_aggregate_matches_definition(self):
        api = self.api()
        a = torch.zeros(4, 4, 2340)
        a[0, :, 0] = 3
        a[1, :, 1] = 2
        a[2, :, 2] = 1
        before = a.clone()
        result = api.aggregate(a, [torch.zeros(2340)])
        self.assertTrue(torch.equal(a, before))
        self.assertTrue(torch.equal(result["flat"], result["u"].sum(0)))
        self.assertTrue(torch.allclose(result["slot"], result["v"].sum(0)))
        self.assertTrue(all(api.norm(v) <= 3.00001 for v in result["v"]))
        saved = result["u"][1].clone()
        result["u"][0].zero_()
        self.assertTrue(torch.equal(saved, result["u"][1]))

    def test_final_slot_norm_can_exceed_flat_norm_and_direction_can_reverse(self):
        api = self.api()
        a = torch.zeros(4, 4, 2)
        a[0, :, 0] = 3.
        a[1:3, :, 0] = -1.375
        result = api.aggregate(a, [torch.zeros(2)])
        self.assertGreater(api.norm(result["slot"]), api.norm(result["flat"]))
        self.assertAlmostEqual(result["metrics"]["flat_slot_cosine"], -1.)
        zeros = api.aggregate(torch.zeros_like(a), [torch.zeros(2)])
        self.assertIsNone(zeros["metrics"]["slot_flat_norm_ratio"])
        self.assertIsNone(zeros["metrics"]["flat_slot_cosine"])

    def test_noise_references_are_analytic_and_scale_by_eight(self):
        api = self.api()
        with patch.object(torch, 'randn', side_effect=AssertionError('no noise draws')):
            ref = api.noise_reference()
        self.assertAlmostEqual(ref["flat_s_sum"] / ref["slot_s_sum"], 8.)
        self.assertAlmostEqual(ref["flat_s_sum"], 2063.145913, places=5)
        self.assertAlmostEqual(ref["slot_s_sum"], 257.893239, places=5)
        result = api.proxy_metrics(10., 2., ref)
        self.assertAlmostEqual(result["proxy_gain_slot_over_flat"], 1.6)
        self.assertIsNone(api.proxy_metrics(0., 1., ref)["proxy_gain_slot_over_flat"])

    def test_batch_is_fixed_across_models_and_global_rng_unchanged(self):
        api = self.api()
        before = random.getstate()
        chosen, state = api.choose_indices(17, 2, 3)
        self.assertEqual((chosen, state), api.choose_indices(17, 2, 3))
        self.assertEqual(len(set(chosen)), 4)
        self.assertEqual(random.getstate(), before)
        self.assertNotEqual(state["initial_state"], state["final_state"])
        with self.assertRaises(ValueError):
            api.choose_indices(3, 2, 3)

    def test_smoke_gate_precedes_other_conditions_and_no_optimizer_allowed(self):
        api = self.api()
        order = api.condition_order()
        self.assertEqual(order[:2], [(0, 0, "original"), (0, 0, "personalized_eps2")])
        self.assertEqual(len(order), 30)
        self.assertEqual(len(set(order)), 30)
        with api.forward_backward_only():
            with self.assertRaises(AssertionError):
                torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=.005)

    def test_entry_has_no_training_updates_noise_or_sampler_calls(self):
        api = self.api()
        import ast
        tree = ast.parse(Path(api.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                self.assertNotIn(name, ('step', 'randn', 'randn_like', 'normal_', 'train_single', 'train_case', 'sample_privim_star', 'build_slots'))


if __name__ == '__main__':
    unittest.main()

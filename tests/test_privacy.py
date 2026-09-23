import unittest

import numpy as np
import torch

from experiments.privacy import (
    FIXED_BATCH_HYPERGEOMETRIC,
    PAPER_BINOMIAL,
    PrivateSGD,
    LEGACY_GLOBAL_DP_NOISE_STREAM,
    multiterm_rdp,
    parameter_noise_displacement_rms,
    rdp_to_dp,
    sensitivity_occurrences,
)


class PrivacyTests(unittest.TestCase):
    @staticmethod
    def _noise_only_update(noise_seed):
        parameter = torch.nn.Parameter(torch.zeros(16))
        optimizer = PrivateSGD(
            [parameter], l2_norm_clip=1.0, noise_multiplier=1.0,
            minibatch_size=1, lr=1.0, noise_seed=noise_seed,
        )
        optimizer.zero_accum_grad()
        optimizer.step()
        return parameter.detach().clone()

    def test_more_noise_reduces_epsilon(self):
        orders = np.arange(2, 32)
        low = multiterm_rdp(orders, 20, 1.0, 256, 10, 16)
        high = multiterm_rdp(orders, 20, 2.0, 256, 10, 16)
        eps_low, _ = rdp_to_dp(orders, low, 1e-5)
        eps_high, _ = rdp_to_dp(orders, high, 1e-5)
        self.assertLess(eps_high, eps_low)

    def test_parameter_noise_displacement_rms(self):
        self.assertAlmostEqual(
            parameter_noise_displacement_rms(100, 2.0, [0.1, 0.2]),
            np.sqrt(100 * 4 * (0.01 + 0.04)),
        )

    def test_private_sgd_updates_after_microbatch_clipping(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = PrivateSGD(
            [parameter], l2_norm_clip=1.0, noise_multiplier=0.0,
            minibatch_size=1, lr=0.1,
        )
        optimizer.zero_accum_grad()
        (parameter * 2).sum().backward()
        optimizer.microbatch_step()
        optimizer.step()
        self.assertAlmostEqual(parameter.item(), 0.9, places=5)
        self.assertAlmostEqual(
            optimizer.last_clipped_gradient_sum_norm, 1.0, places=5
        )
        self.assertEqual(optimizer.last_noise_norm, 0.0)
        self.assertAlmostEqual(
            optimizer.last_noisy_gradient_sum_norm, 1.0, places=5
        )

    def test_zero_noise_clipping_averages_clipped_subgraph_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = PrivateSGD(
            [parameter], l2_norm_clip=1.0, noise_multiplier=0.0,
            minibatch_size=2, lr=0.1,
        )
        optimizer.zero_accum_grad()
        (parameter * 2).sum().backward()
        optimizer.microbatch_step()
        optimizer.zero_microbatch_grad()
        (parameter * -0.5).sum().backward()
        optimizer.microbatch_step()
        optimizer.step()
        # clipped gradients are +1 and -0.5; their batch mean is +0.25.
        self.assertAlmostEqual(parameter.item(), 0.975, places=5)
        self.assertEqual(optimizer.last_microbatch_norms, [2.0, 0.5])
        self.assertAlmostEqual(optimizer.last_clip_fraction, 0.5)
        self.assertAlmostEqual(
            optimizer.last_clipped_gradient_sum_norm, 0.5, places=5
        )
        self.assertEqual(optimizer.last_noise_norm, 0.0)
        self.assertAlmostEqual(
            optimizer.last_noisy_gradient_sum_norm, 0.5, places=5
        )

    def test_accountant_and_optimizer_sensitivity_are_explicit(self):
        self.assertEqual(
            sensitivity_occurrences(
                FIXED_BATCH_HYPERGEOMETRIC, 1111, 80, 16
            ),
            16,
        )
        self.assertEqual(
            sensitivity_occurrences(PAPER_BINOMIAL, 1111, 80, 16),
            80,
        )

    def test_dp_noise_seed_is_deterministic_and_independent_of_global_rng(self):
        torch.manual_seed(1)
        torch.randn(100)
        first = self._noise_only_update(17)
        torch.manual_seed(999)
        torch.randn(7)
        second = self._noise_only_update(17)
        self.assertTrue(torch.equal(first, second))

    def test_different_dp_noise_seeds_change_the_update(self):
        first = self._noise_only_update(17)
        second = self._noise_only_update(18)
        self.assertFalse(torch.equal(first, second))

    def test_noise_diagnostics_separate_signal_and_noise_norms(self):
        parameter = torch.nn.Parameter(torch.zeros(16))
        optimizer = PrivateSGD(
            [parameter], l2_norm_clip=1.0, noise_multiplier=1.0,
            minibatch_size=1, lr=1.0, noise_seed=17,
        )
        optimizer.zero_accum_grad()
        optimizer.step()
        self.assertEqual(optimizer.last_clipped_gradient_sum_norm, 0.0)
        self.assertGreater(optimizer.last_noise_norm, 0.0)
        self.assertAlmostEqual(
            optimizer.last_noisy_gradient_sum_norm,
            optimizer.last_noise_norm,
            places=6,
        )

    def test_dp_noise_generator_does_not_advance_global_rng(self):
        torch.manual_seed(23)
        expected = torch.randn(8)
        torch.manual_seed(23)
        self._noise_only_update(17)
        actual = torch.randn(8)
        self.assertTrue(torch.equal(expected, actual))

    def test_legacy_global_noise_matches_and_advances_global_rng(self):
        torch.manual_seed(23)
        expected_update = -torch.randn(16)
        expected_after = torch.randn(8)

        torch.manual_seed(23)
        parameter = torch.nn.Parameter(torch.zeros(16))
        optimizer = PrivateSGD(
            [parameter], l2_norm_clip=1.0, noise_multiplier=1.0,
            minibatch_size=1, lr=1.0, noise_seed=999,
            noise_stream=LEGACY_GLOBAL_DP_NOISE_STREAM,
        )
        optimizer.zero_accum_grad()
        optimizer.step()
        actual_after = torch.randn(8)

        self.assertTrue(torch.equal(parameter.detach(), expected_update))
        self.assertTrue(torch.equal(actual_after, expected_after))

    def test_named_accountants_are_finite_and_distinct(self):
        orders = np.arange(2, 16)
        fixed = multiterm_rdp(
            orders, 10, 2.0, 100, 20, 16,
            accountant=FIXED_BATCH_HYPERGEOMETRIC,
        )
        paper = multiterm_rdp(
            orders, 10, 2.0, 100, 20, 16,
            accountant=PAPER_BINOMIAL,
        )
        self.assertTrue(np.isfinite(fixed).all())
        self.assertTrue(np.isfinite(paper).all())
        self.assertFalse(np.allclose(fixed, paper))


if __name__ == "__main__":
    unittest.main()

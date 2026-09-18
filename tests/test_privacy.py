import unittest

import numpy as np
import torch

from experiments.privacy import (
    FIXED_BATCH_HYPERGEOMETRIC,
    PAPER_BINOMIAL,
    PrivateSGD,
    multiterm_rdp,
    rdp_to_dp,
    sensitivity_occurrences,
)


class PrivacyTests(unittest.TestCase):
    def test_more_noise_reduces_epsilon(self):
        orders = np.arange(2, 32)
        low = multiterm_rdp(orders, 20, 1.0, 256, 10, 16)
        high = multiterm_rdp(orders, 20, 2.0, 256, 10, 16)
        eps_low, _ = rdp_to_dp(orders, low, 1e-5)
        eps_high, _ = rdp_to_dp(orders, high, 1e-5)
        self.assertLess(eps_high, eps_low)

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

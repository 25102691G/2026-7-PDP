"""Meaningful B0 invariants; statistical checks are diagnostics, not DP proofs."""

import math
import random
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from unittest.mock import patch

from pdp.accounting import RangePrivacy, epsilon_upper
from pdp.discrete_noise import (
    DiscreteGaussian, _exponential_coin, _gaussian_integer,
)
from pdp.lattice import (
    project_integer_ball, quantize_gradient, sum_contributions,
)


class IntegerRangeTests(unittest.TestCase):
    def test_clipping_bound_including_extreme_and_nonfinite_gradients(self):
        rng = random.Random(12)
        vectors = [
            [], [0.0, 0.0], [1e308, -1e308, 1e-308],
            [float("nan"), 1], [float("inf"), 1],
        ] + [[rng.uniform(-100, 100) for _ in range(37)] for _ in range(100)]
        for values in vectors:
            result = quantize_gradient(values, radius=65536, grid_bits=16)
            self.assertEqual(len(result), len(values))
            self.assertLessEqual(sum(v*v for v in result), 65536**2)
        self.assertEqual(
            quantize_gradient([float("inf"), 1], radius=10, grid_bits=4),
            [0, 0],
        )

    def test_small_gradients_preserve_magnitude_and_sign(self):
        self.assertEqual(
            quantize_gradient([0.125, -0.25], radius=32, grid_bits=4),
            [2, -4],
        )

    def test_integer_projection_does_not_rely_on_floating_norm(self):
        values = [10**1000, -(10**1000)]
        result = project_integer_ball(values, radius=100)
        self.assertLessEqual(sum(v*v for v in result), 100**2)
        self.assertGreater(result[0], 0)
        self.assertLess(result[1], 0)

    def test_full_replacement_can_attain_2BC_and_empty_is_padded(self):
        positive = sum_contributions(
            [[3, 4]] * 16, radius=5, batch_size=16, dimension=2,
        )
        negative = sum_contributions(
            [[-3, -4]] * 16, radius=5, batch_size=16, dimension=2,
        )
        self.assertEqual(
            sum((a-b)**2 for a, b in zip(positive, negative)), (2*16*5)**2,
        )
        self.assertEqual(
            sum_contributions([], radius=5, batch_size=16, dimension=2),
            [0, 0],
        )
        with self.assertRaises(ValueError):
            sum_contributions([[1]]*17, radius=5, batch_size=16, dimension=1)

    def test_calibration_is_independent_of_graph_size_and_certified_outward(self):
        for epsilon in ["1", "2", "4"]:
            plan = RangePrivacy(epsilon=epsilon, delta="0.00001")
            scale = plan.calibrate()
            certificate = plan.certificate(scale)
            self.assertLessEqual(Decimal(certificate["epsilon_upper"]), Decimal(epsilon))
            self.assertGreater(
                epsilon_upper(plan.rho(scale-1), Decimal(plan.delta)),
                Decimal(epsilon),
            )
            with localcontext() as context:
                context.prec = 130
                rho = Decimal(plan.rho(scale).numerator)/Decimal(plan.rho(scale).denominator)
                reference = rho + 2*(rho * -Decimal(plan.delta).ln()).sqrt()
            self.assertGreaterEqual(Decimal(certificate["epsilon_upper"]), reference)
            for forbidden in ["train_nodes", "container_size", "seed", "node_hash"]:
                self.assertNotIn(forbidden, certificate)

    def test_invalid_public_configuration_is_rejected_before_data(self):
        for arguments in [
            {"epsilon": "NaN"}, {"epsilon": "0"}, {"delta": "1"},
            {"clip_norm": "0"}, {"clip_norm": "0.000000000000001"},
            {"batch_size": 0}, {"steps": True}, {"grid_bits": 41},
        ]:
            with self.assertRaises(ValueError):
                RangePrivacy(**arguments)

    def test_noise_budget_changes_with_actual_T_and_not_mean_scaling(self):
        short = RangePrivacy(steps=10, batch_size=1, grid_bits=0)
        long = RangePrivacy(steps=100, batch_size=1, grid_bits=0)
        batched = RangePrivacy(steps=100, batch_size=16, grid_bits=0)
        self.assertGreater(long.calibrate(), short.calibrate())
        self.assertGreaterEqual(batched.calibrate(), 16*long.calibrate()-16)
        with self.assertRaises(ValueError):
            long.certificate(short.calibrate())


class ExactNoiseTests(unittest.TestCase):
    def test_exponential_coin_boundaries_and_known_probabilities(self):
        rng = random.Random(19)
        self.assertTrue(_exponential_coin(Fraction(0), rng.randrange))
        for exponent in [Fraction(1, 2), Fraction(1), Fraction(7, 2)]:
            count = sum(_exponential_coin(exponent, rng.randrange) for _ in range(12000))
            self.assertLess(abs(count/12000-math.exp(-float(exponent))), 0.02)

    def test_integer_gaussian_matches_pmf_at_small_scale(self):
        rng = random.Random(67)
        count = 20000
        samples = [_gaussian_integer(1, rng.randrange) for _ in range(count)]
        normalizer = sum(math.exp(-x*x/2) for x in range(-12, 13))
        for value in [-2, -1, 0, 1, 2]:
            expected = math.exp(-value*value/2)/normalizer
            self.assertLess(abs(samples.count(value)/count-expected), 0.015)
        self.assertLess(abs(sum(samples)/count), 0.04)

    def test_large_scale_uses_integer_arithmetic_and_production_source(self):
        with patch("pdp.discrete_noise.secrets.randbelow",
                   side_effect=random.Random(9).randrange) as uniform:
            samples = DiscreteGaussian(10**20).sample(6)
        self.assertTrue(uniform.called)
        self.assertTrue(all(isinstance(value, int) for value in samples))
        self.assertTrue(any(abs(value) > 2**53 for value in samples))
        with self.assertRaises(TypeError):
            DiscreteGaussian(3, seed=0)

    def test_bidirectional_discrete_rdp_is_below_proved_bound(self):
        # Directly sum a rapidly convergent infinite-series truncation only
        # as a check against implementation/model mismatches.
        sigma = 3
        for alpha in [1.5, 2, 3.25]:
            normalizer = sum(math.exp(-x*x/(2*sigma*sigma)) for x in range(-80, 81))
            for left, right in [(0, 2), (2, 0)]:
                moment = sum(
                    math.exp(
                        (-alpha*(x-left)**2+(alpha-1)*(x-right)**2)
                        /(2*sigma*sigma)
                    ) / normalizer for x in range(-80, 81)
                )
                divergence = math.log(moment)/(alpha-1)
                self.assertLessEqual(divergence, alpha*4/(2*sigma*sigma)+1e-12)


if __name__ == "__main__":
    unittest.main()

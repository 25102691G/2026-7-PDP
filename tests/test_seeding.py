import unittest

from experiments.seeding import resolve_round_seeds, seed_protocol


class SeedProtocolTests(unittest.TestCase):
    def test_fixed_sampling_decouples_round_randomness(self):
        first = resolve_round_seeds(
            round_index=0, model_seed=10, sampling_seed=20,
            model_seed_mode="per-round", batch_seed=30,
            batch_seed_mode="per-round", sampling_seed_mode="fixed",
            dp_noise_seed=40, dp_noise_seed_mode="per-round",
        )
        second = resolve_round_seeds(
            round_index=1, model_seed=10, sampling_seed=20,
            model_seed_mode="per-round", batch_seed=30,
            batch_seed_mode="per-round", sampling_seed_mode="fixed",
            dp_noise_seed=40, dp_noise_seed_mode="per-round",
        )
        self.assertEqual((first.model, second.model), (10, 11))
        self.assertEqual((first.sampling, second.sampling), (20, 20))
        self.assertEqual((first.batch, second.batch), (30, 31))
        self.assertEqual((first.dp_noise, second.dp_noise), (40, 41))

    def test_per_round_sampling_changes_only_by_round_offset(self):
        resolved = resolve_round_seeds(
            round_index=3, model_seed=10, sampling_seed=20,
            model_seed_mode="per-round", batch_seed=30,
            batch_seed_mode="per-round", sampling_seed_mode="per-round",
            dp_noise_seed=40, dp_noise_seed_mode="fixed",
        )
        self.assertEqual(resolved.as_dict(), {
            "model": 13,
            "sampling": 23,
            "batch": 33,
            "dp_noise": 40,
        })

    def test_protocol_is_manifest_serializable(self):
        self.assertEqual(
            seed_protocol(
                model_seed=1, model_seed_mode="per-round",
                sampling_seed=2, batch_seed=3,
                batch_seed_mode="per-round", sampling_seed_mode="fixed",
                dp_noise_seed=4, dp_noise_seed_mode="fixed",
            ),
            {
                "model_seed_base": 1,
                "model_seed_mode": "per-round",
                "sampling_seed_base": 2,
                "sampling_seed_mode": "fixed",
                "batch_seed_base": 3,
                "batch_seed_mode": "per-round",
                "dp_noise_seed_base": 4,
                "dp_noise_seed_mode": "fixed",
            },
        )

    def test_sampling_only_mode_holds_model_and_batch_fixed(self):
        resolved = resolve_round_seeds(
            round_index=4, model_seed=10, model_seed_mode="fixed",
            sampling_seed=20, sampling_seed_mode="per-round",
            batch_seed=30, batch_seed_mode="fixed",
            dp_noise_seed=40, dp_noise_seed_mode="fixed",
        )
        self.assertEqual(resolved.as_dict(), {
            "model": 10,
            "sampling": 24,
            "batch": 30,
            "dp_noise": 40,
        })

    def test_dp_noise_only_mode_changes_only_noise_seed(self):
        resolved = resolve_round_seeds(
            round_index=4, model_seed=10, model_seed_mode="fixed",
            sampling_seed=20, sampling_seed_mode="fixed",
            batch_seed=30, batch_seed_mode="fixed",
            dp_noise_seed=40, dp_noise_seed_mode="per-round",
        )
        self.assertEqual(resolved.as_dict(), {
            "model": 10,
            "sampling": 20,
            "batch": 30,
            "dp_noise": 44,
        })

    def test_continuous_sampling_uses_one_base_stream_across_rounds(self):
        first = resolve_round_seeds(
            round_index=0, model_seed=10, model_seed_mode="fixed",
            sampling_seed=20, sampling_seed_mode="continuous",
            batch_seed=30, batch_seed_mode="fixed",
            dp_noise_seed=40, dp_noise_seed_mode="fixed",
        )
        later = resolve_round_seeds(
            round_index=4, model_seed=10, model_seed_mode="fixed",
            sampling_seed=20, sampling_seed_mode="continuous",
            batch_seed=30, batch_seed_mode="fixed",
            dp_noise_seed=40, dp_noise_seed_mode="fixed",
        )
        self.assertEqual(first.sampling, 20)
        self.assertEqual(later.sampling, 20)
        protocol = seed_protocol(
            model_seed=10, model_seed_mode="fixed",
            sampling_seed=20, sampling_seed_mode="continuous",
            batch_seed=30, batch_seed_mode="fixed",
            dp_noise_seed=40, dp_noise_seed_mode="fixed",
        )
        self.assertEqual(protocol["sampling_seed_mode"], "continuous")


if __name__ == "__main__":
    unittest.main()

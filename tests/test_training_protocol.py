import unittest

import torch

from experiments.config import released_compatibility_overrides
from experiments.train import (
    apply_parameter_ema,
    training_step_count,
    update_parameter_ema,
)


class TrainingProtocolTests(unittest.TestCase):
    def test_parameter_ema_is_detached_and_can_be_applied(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema = update_parameter_ema(None, model, decay=0.5)
        with torch.no_grad():
            model.weight.fill_(3.0)
        ema = update_parameter_ema(ema, model, decay=0.5)
        self.assertEqual(float(ema["weight"]), 2.0)
        with torch.no_grad():
            model.weight.fill_(9.0)
        apply_parameter_ema(ema, model)
        self.assertEqual(float(model.weight), 2.0)

    def test_fixed_iterations_are_actual_optimizer_steps(self):
        self.assertEqual(
            training_step_count(266, 16, 500, "fixed-iterations"),
            500,
        )

    def test_released_epochs_traverse_every_batch(self):
        self.assertEqual(
            training_step_count(32, 16, 20, "released-epochs"),
            40,
        )
        self.assertEqual(
            training_step_count(266, 16, 20, "released-epochs"),
            340,
        )

    def test_released_epoch_update_cap_only_changes_actual_steps(self):
        self.assertEqual(
            training_step_count(
                320, 16, 20, "released-epochs", optimizer_update_cap=320
            ),
            320,
        )
        self.assertEqual(
            training_step_count(
                110, 16, 20, "released-epochs", optimizer_update_cap=120
            ),
            120,
        )
        with self.assertRaisesRegex(ValueError, "only supported"):
            training_step_count(
                320, 16, 500, "fixed-iterations", optimizer_update_cap=320
            )

    def test_released_profile_preserves_audited_randomness_and_accountant(self):
        profile = released_compatibility_overrides()
        self.assertEqual(profile["iterations"], 20)
        self.assertEqual(profile["privacy_accountant"], "fixed_batch_hypergeometric")
        self.assertEqual(profile["sampling_seed_mode"], "continuous")
        self.assertEqual(profile["model_seed_mode"], "fixed")
        self.assertEqual(profile["batch_seed_mode"], "fixed")
        self.assertEqual(profile["dp_noise_seed_mode"], "fixed")


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_reproduction import completed_models


class ReproductionTests(unittest.TestCase):
    def test_resume_only_counts_matching_iteration_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "email" / "PrivIM+" / "eps_10000" / "n_40"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            for seed, iterations in ((0, 50), (1, 500)):
                checkpoint = model_dir / f"model_GRAT3_d_1_seed_{seed}_round_{seed}.pt"
                checkpoint.touch()
                checkpoint.with_suffix(".json").write_text(
                    json.dumps({"iterations": iterations}), encoding="utf-8"
                )

            self.assertEqual(
                completed_models(root, "email", "PrivIM+", "eps_10000", 500),
                1,
            )
            self.assertEqual(
                completed_models(root, "email", "PrivIM+", "eps_10000", 50),
                1,
            )

    def test_resume_checks_learning_rate_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "email" / "PrivIM+" / "eps_10000" / "n_40"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_GRAT3_d_1_seed_0_round_0.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").write_text(
                json.dumps({
                    "iterations": 500,
                    "learning_rate": 0.005,
                    "learning_rate_schedule": {
                        "gamma": 0.99,
                        "step_size_updates": 5,
                    },
                }),
                encoding="utf-8",
            )

            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_10000", 500,
                    0.005, 0.99, 5,
                ),
                1,
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_10000", 500,
                    0.005, 1.0, 5,
                ),
                0,
            )

    def test_resume_checks_seed_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "email" / "PrivIM+" / "eps_10000" / "n_40"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_GRAT3_d_1_seed_0_round_0.pt"
            checkpoint.touch()
            protocol = {
                "model_seed_base": 0,
                "sampling_seed_base": 7,
                "sampling_seed_mode": "fixed",
                "batch_seed_base": 20,
            }
            checkpoint.with_suffix(".json").write_text(
                json.dumps({
                    "iterations": 500,
                    "experiment": {"seed_protocol": protocol},
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_10000", 500,
                    expected_seed_protocol=protocol,
                ),
                1,
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_10000", 500,
                    expected_seed_protocol={**protocol, "sampling_seed_base": 8},
                ),
                0,
            )

    def test_resume_checks_model_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "email" / "PrivIM+" / "eps_10000" / "n_40"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_GRAT3_d_1_seed_0_round_0.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").write_text(
                json.dumps({
                    "iterations": 500,
                    "model_initialization": "structural_preserving",
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_10000", 500,
                    model_initialization="structural_preserving",
                ),
                1,
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_10000", 500,
                    model_initialization="released_default",
                ),
                0,
            )

    def test_resume_checks_non_private_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "facebook" / "PrivIM+" / "eps_10000" / "n_40"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_GRAT3_d_1_seed_0_round_0.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").write_text(
                json.dumps({
                    "iterations": 500,
                    "non_private_clipping": True,
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                completed_models(
                    root, "facebook", "PrivIM+", "eps_10000", 500,
                    non_private_clipping=True,
                ),
                1,
            )
            self.assertEqual(
                completed_models(
                    root, "facebook", "PrivIM+", "eps_10000", 500,
                    non_private_clipping=False,
                ),
                0,
            )

    def test_resume_checks_feature_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "facebook" / "PrivIM+" / "eps_10000" / "n_60"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_GRAT3_d_1_seed_0_round_0.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").write_text(
                json.dumps({
                    "iterations": 500,
                    "feature_protocol": "degree",
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                completed_models(
                    root, "facebook", "PrivIM+", "eps_10000", 500,
                    feature_protocol="degree",
                ),
                1,
            )
            self.assertEqual(
                completed_models(
                    root, "facebook", "PrivIM+", "eps_10000", 500,
                    feature_protocol="ones",
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()

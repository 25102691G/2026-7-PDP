import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.run_reproduction import completed_models, subprocess_environment


class ReproductionTests(unittest.TestCase):
    def test_loss_penalty_is_forwarded(self):
        result = subprocess.run(
            [
                sys.executable,
                "experiments/run_reproduction.py",
                "--dataset", "hepph",
                "--only-non-private",
                "--loss-penalty", "16",
                "--skip-prepare",
                "--device", "cpu",
                "--dry-run",
                "--run-id", "test-loss-penalty",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--loss-penalty 16.0", result.stdout)

    def test_strict_optimizer_momentum_is_forwarded(self):
        result = subprocess.run(
            [
                sys.executable,
                "experiments/run_reproduction.py",
                "--dataset", "email",
                "--epsilon", "4",
                "--settings", "PrivIM+",
                "--optimizer-momentum", "0.9",
                "--skip-prepare",
                "--skip-non-private",
                "--device", "cpu",
                "--dry-run",
                "--run-id", "test-strict-momentum",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--optimizer-momentum 0.9", result.stdout)

    def test_strict_ema_and_batch_size_are_forwarded(self):
        result = subprocess.run(
            [
                sys.executable,
                "experiments/run_reproduction.py",
                "--dataset", "email",
                "--epsilon", "4",
                "--settings", "PrivIM+",
                "--batch-size", "8",
                "--parameter-averaging", "ema",
                "--ema-decay", "0.95",
                "--skip-prepare",
                "--skip-non-private",
                "--device", "cpu",
                "--dry-run",
                "--run-id", "test-strict-ema",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--batchsize 8", result.stdout)
        self.assertIn("--parameter-averaging ema", result.stdout)
        self.assertIn("--ema-decay 0.95", result.stdout)

    def test_released_compatibility_rejects_ema(self):
        result = subprocess.run(
            [
                sys.executable,
                "experiments/run_reproduction.py",
                "--protocol-profile", "released-compatibility",
                "--parameter-averaging", "ema",
                "--dataset", "email",
                "--epsilon", "4",
                "--settings", "PrivIM+",
                "--skip-prepare",
                "--skip-non-private",
                "--device", "cpu",
                "--dry-run",
                "--run-id", "test-released-ema-rejected",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "released-compatibility requires --parameter-averaging none",
            result.stderr,
        )

    def test_released_compatibility_profile_is_explicit_in_dry_run(self):
        result = subprocess.run(
            [
                sys.executable,
                "experiments/run_reproduction.py",
                "--protocol-profile", "released-compatibility",
                "--dataset", "email",
                "--epsilon", "4",
                "--settings", "PrivIM+",
                "--iterations", "999",
                "--lr-decay", "1",
                "--skip-prepare",
                "--skip-non-private",
                "--device", "cpu",
                "--dry-run",
                "--run-id", "test-released-profile",
            ],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        command = result.stdout
        self.assertIn("--n_iteration 20", command)
        self.assertIn("--training-protocol released-epochs", command)
        self.assertIn("--privacy-accounting-steps 20", command)
        self.assertIn("--privacy-accountant fixed_batch_hypergeometric", command)
        self.assertIn("--optimizer-momentum 0.9", command)
        self.assertIn("--sensitivity-noise-factor 2.0", command)
        self.assertIn("--base-occurrence-bound 10", command)
        self.assertIn("--r 3", command)
        self.assertIn("--sampling-seed-mode continuous", command)
        self.assertIn("--model-seed-mode fixed", command)
        self.assertIn("--dp-noise-seed-mode fixed", command)
        self.assertIn("--sampling-protocol paper", command)

    def test_released_legacy_sampler_is_explicit_and_profile_gated(self):
        command = [
            sys.executable,
            "experiments/run_reproduction.py",
            "--protocol-profile", "released-compatibility",
            "--sampling-protocol", "released-legacy",
            "--dataset", "facebook",
            "--epsilon", "4",
            "--settings", "PrivIM+",
            "--skip-prepare",
            "--skip-non-private",
            "--device", "cpu",
            "--dry-run",
            "--run-id", "test-released-legacy-sampler",
        ]
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--sampling-protocol released-legacy", result.stdout)

        command[command.index("released-compatibility")] = "strict"
        rejected = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(
            "released-legacy requires --protocol-profile released-compatibility",
            rejected.stderr,
        )

    def test_gpu_environment_is_isolated_and_pinned(self):
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        environment = subprocess_environment(2)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), original)

    def test_compatibility_update_cap_is_forwarded_and_profile_gated(self):
        command = [
            sys.executable,
            "experiments/run_reproduction.py",
            "--protocol-profile", "released-compatibility",
            "--sampling-protocol", "released-legacy",
            "--compatibility-update-cap", "320",
            "--dataset", "facebook",
            "--epsilon", "4",
            "--settings", "PrivIM+",
            "--skip-prepare",
            "--skip-non-private",
            "--device", "cpu",
            "--dry-run",
            "--run-id", "test-compatibility-update-cap",
        ]
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--optimizer-update-cap 320", result.stdout)

        command[command.index("released-compatibility")] = "strict"
        rejected = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(
            "--compatibility-update-cap requires",
            rejected.stderr,
        )

    def test_legacy_global_dp_noise_is_forwarded_and_profile_gated(self):
        command = [
            sys.executable,
            "experiments/run_reproduction.py",
            "--protocol-profile", "released-compatibility",
            "--dp-noise-stream", "legacy-global",
            "--dataset", "facebook",
            "--epsilon", "4",
            "--settings", "PrivIM+",
            "--skip-prepare",
            "--skip-non-private",
            "--device", "cpu",
            "--dry-run",
            "--run-id", "test-legacy-global-noise",
        ]
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--dp-noise-stream legacy-global", result.stdout)

        command[command.index("released-compatibility")] = "strict"
        rejected = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn(
            "--dp-noise-stream legacy-global requires",
            rejected.stderr,
        )

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

    def test_resume_checks_optimizer_momentum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "email" / "PrivIM+" / "eps_4" / "n_40"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_GRAT3_d_1_seed_0_round_0.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").write_text(
                json.dumps({"iterations": 200, "momentum": 0.9}),
                encoding="utf-8",
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_4", 200,
                    optimizer_momentum=0.9,
                ),
                1,
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_4", 200,
                    optimizer_momentum=0.0,
                ),
                0,
            )

    def test_resume_checks_loss_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "hepph" / "PrivIM+" / "eps_10000" / "n_60"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_GRAT3_d_1_seed_0_round_0.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").write_text(
                json.dumps({"iterations": 500, "loss_lambda": 0.5}),
                encoding="utf-8",
            )
            self.assertEqual(
                completed_models(
                    root, "hepph", "PrivIM+", "eps_10000", 500,
                    loss_penalty=0.5,
                ),
                1,
            )
            self.assertEqual(
                completed_models(
                    root, "hepph", "PrivIM+", "eps_10000", 500,
                    loss_penalty=1.0,
                ),
                0,
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
                "model_seed_mode": "fixed",
                "sampling_seed_base": 7,
                "sampling_seed_mode": "fixed",
                "batch_seed_base": 20,
                "batch_seed_mode": "fixed",
                "dp_noise_seed_base": 30,
                "dp_noise_seed_mode": "per-round",
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
                    expected_seed_protocol={**protocol, "dp_noise_seed_base": 31},
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

    def test_resume_checks_protocol_profile_and_configured_units(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = (
                root / "email" / "PrivIM+" / "eps_4" / "n_40"
                / "Ng_4" / "model"
            )
            model_dir.mkdir(parents=True)
            checkpoint = model_dir / "model_GRAT3_d_1_seed_0_round_0.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").write_text(
                json.dumps({
                    "iterations": 40,
                    "configured_training_units": 20,
                    "protocol_profile": "released-compatibility",
                    "sampling_protocol": "released-legacy",
                    "optimizer_update_cap": 320,
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_4", 20,
                    protocol_profile="released-compatibility",
                    sampling_protocol="released-legacy",
                    optimizer_update_cap=320,
                ),
                1,
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_4", 20,
                    protocol_profile="strict",
                    sampling_protocol="released-legacy",
                    optimizer_update_cap=320,
                ),
                0,
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_4", 20,
                    protocol_profile="released-compatibility",
                    sampling_protocol="paper",
                    optimizer_update_cap=320,
                ),
                0,
            )
            self.assertEqual(
                completed_models(
                    root, "email", "PrivIM+", "eps_4", 20,
                    protocol_profile="released-compatibility",
                    sampling_protocol="released-legacy",
                    optimizer_update_cap=120,
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()

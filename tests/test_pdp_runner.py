import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("DGLBACKEND", "pytorch")
import igraph
import torch

from pdp.evaluation import evaluate_release
from pdp.run import load_protocol, planned_cells, run_protocol, write_summaries
from pdp.training import ParameterRelease, TrainingConfig, initialize_model


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_canonical_plan_has_all_five_seeds_and_shared_public_parameters(self):
        protocol = load_protocol(ROOT/"configs/pdp/pilot-v1.json")
        cells = planned_cells(protocol)
        self.assertEqual(len(cells), 60)
        self.assertEqual(len({cell["id"] for cell in cells}), 60)
        for dataset in protocol["datasets"]:
            for method in protocol["methods"]:
                self.assertEqual(
                    [cell["model_seed"] for cell in cells
                     if cell["dataset"] == dataset and cell["method"] == method],
                    [0, 1, 2, 3, 4],
                )

    def test_public_star_evaluation_reuses_celf_and_does_not_change_target(self):
        graph = igraph.Graph(n=60, edges=[(0, i) for i in range(1, 60)], directed=True)
        before = graph.get_edgelist()
        config = TrainingConfig(hidden_dim=4)
        model = initialize_model(config)
        release = ParameterRelease(
            {key: torch.zeros_like(value) for key, value in model.state_dict().items()},
            {"privacy_guarantee": "data_independent"},
        )
        result = evaluate_release(release, config, graph)
        self.assertEqual(result["celf_denominator"], 60)
        self.assertEqual(result["spread"], 60)
        self.assertEqual(result["coverage_ratio_percent"], 100)
        self.assertEqual(result["seed_local_ids"], list(range(50)))
        self.assertEqual(graph.get_edgelist(), before)

    def test_failed_seed_is_retained_and_prevents_success_only_summary(self):
        manifest = {"cells": [
            {"dataset": "email", "method": "uniform_dp", "model_seed": 0,
             "optimizer_steps": 100, "status": "completed",
             "metrics": {"coverage_ratio_percent": 99}},
            {"dataset": "email", "method": "uniform_dp", "model_seed": 1,
             "optimizer_steps": 100, "status": "failed"},
        ]}
        with tempfile.TemporaryDirectory() as root:
            write_summaries(Path(root), manifest)
            summary = json.loads((Path(root)/"summary.json").read_text())[0]
            with (Path(root)/"records.csv").open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["status"], "failed")
        self.assertIsNone(summary["coverage_mean"])
        self.assertFalse(summary["all_runs_complete"])

    def test_end_to_end_and_resume_do_not_repeat_completed_private_calls(self):
        protocol = json.loads((ROOT/"configs/pdp/pilot-v1.json").read_text())
        protocol["datasets"] = ["email"]
        protocol["methods"] = ["public_init", "uniform_dp"]
        protocol["model_seeds"] = [0]
        protocol["training"].update({
            "private_steps": 1, "batch_size": 2, "hidden_dim": 4,
            "subgraph_size_by_dataset": {"email": 2},
        })
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            prepared = root/"prepared"/"email"
            prepared.mkdir(parents=True)
            (prepared/"metadata.json").write_text(json.dumps({
                "train_nodes": 6, "test_nodes": 60, "directed": True,
            }))
            for split, count, offset in [("train", 6, 0), ("test", 60, 1000)]:
                (prepared/f"email_{split}_edges.txt").write_text(
                    "".join(f"{i} {i+1}\n" for i in range(count-1))
                )
                (prepared/f"{split}_nodes.csv").write_text(
                    "original_id,local_id\n"+
                    "".join(f"{i+offset},{i}\n" for i in range(count))
                )
            path = root/"protocol.json"
            path.write_text(json.dumps(protocol))
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_protocol(
                    path, "test", output_root=root/"runs",
                    prepared_root=root/"prepared",
                )
            manifest = json.loads((result/"manifest.json").read_text())
            self.assertTrue(all(cell["status"] == "completed" for cell in manifest["cells"]))
            self.assertEqual(len(list((result/"trusted"/"models").iterdir())), 2)
            # Simulate interruption after writing a complete model but before
            # its evaluation/status update. Recovery must not call training.
            manifest["cells"][1]["status"] = "running"
            (result/"manifest.json").write_text(json.dumps(manifest))
            with patch("pdp.run.train_b0", side_effect=AssertionError("must not retrain")):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_protocol(
                        path, "test", resume=True, output_root=root/"runs",
                        prepared_root=root/"prepared",
                    )
            recovered = json.loads((result/"manifest.json").read_text())
            self.assertIn("completed_at_utc", recovered)
            self.assertTrue(all(cell["status"] == "completed" for cell in recovered["cells"]))
            protocol["training"]["private_steps"] = 2
            path.write_text(json.dumps(protocol))
            with self.assertRaises(ValueError):
                run_protocol(path, "test", resume=True, output_root=root/"runs",
                             prepared_root=root/"prepared")

    def test_protocol_rejects_duplicate_seeds_and_unfair_uniform_budget(self):
        protocol = json.loads((ROOT/"configs/pdp/pilot-v1.json").read_text())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"protocol.json"
            duplicate = copy.deepcopy(protocol)
            duplicate["model_seeds"] = [0, 0]
            path.write_text(json.dumps(duplicate))
            with self.assertRaises(ValueError):
                load_protocol(path)
            unfair = copy.deepcopy(protocol)
            unfair["base_epsilon"]["uniform_dp"] = "4"
            path.write_text(json.dumps(unfair))
            with self.assertRaises(ValueError):
                load_protocol(path)

    def test_stable_slot_runner_reuses_controls_and_recovers_without_retraining(self):
        protocol = json.loads((ROOT/"configs/pdp/pilot-v1.json").read_text())
        protocol.update({"datasets": ["email"], "methods": ["public_init", "non_private"],
                         "model_seeds": [0]})
        protocol["training"].update({
            "private_steps": 1, "non_private_steps": 1, "batch_size": 2,
            "hidden_dim": 4, "subgraph_size_by_dataset": {"email": 2},
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root/"prepared"/"email"
            prepared.mkdir(parents=True)
            (prepared/"metadata.json").write_text(json.dumps({
                "train_nodes": 8, "test_nodes": 60, "directed": True,
            }))
            for split, count in [("train", 8), ("test", 60)]:
                (prepared/f"email_{split}_edges.txt").write_text(
                    "".join(f"{i} {i+1}\n" for i in range(count-1))
                )
                (prepared/f"{split}_nodes.csv").write_text(
                    "original_id,local_id\n"+"".join(f"{i},{i}\n" for i in range(count))
                )
            source = root/"protocol.json"
            source.write_text(json.dumps(protocol))
            with contextlib.redirect_stdout(io.StringIO()):
                reference = run_protocol(source, "reference", output_root=root/"runs",
                                         prepared_root=root/"prepared")
            original = {p.relative_to(reference): p.read_bytes()
                        for p in reference.rglob("*") if p.is_file()}
            protocol.update({
                "methods": ["public_init", "non_private", "uniform_dp", "pdp_slots"],
                "reuse_controls_from": "reference",
                "stable_slots": {
                    "bins_per_round": 2, "rounds": 1, "microbatch_size": 1,
                    "appearance_limits_by_method": {"uniform_dp": [1, 1, 1], "pdp_slots": [1, 1, 1]},
                    "uniform_methods": ["uniform_dp"],
                },
            })
            source.write_text(json.dumps(protocol))
            with patch("pdp.run.train_non_private", side_effect=AssertionError("must reuse")):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run_protocol(source, "slots", output_root=root/"runs",
                                          prepared_root=root/"prepared")
            manifest = json.loads((result/"manifest.json").read_text())
            self.assertTrue(all(c["status"] == "completed" for c in manifest["cells"]))
            self.assertEqual(sum("reused_from" in c for c in manifest["cells"]), 2)
            self.assertTrue(all((reference/p).read_bytes() == data for p, data in original.items()))
            self.assertEqual((result/"protocol.source.json").read_bytes(), source.read_bytes())
            self.assertTrue((result/"trusted/proofs/pdp_stable_slots_design.md").is_file())
            manifest["cells"][-1]["status"] = "running"
            (result/"manifest.json").write_text(json.dumps(manifest))
            with patch("pdp.run.train_slots", side_effect=AssertionError("must recover saved model")):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_protocol(source, "slots", resume=True, output_root=root/"runs",
                                 prepared_root=root/"prepared")
            # A different control configuration must never be silently reused.
            protocol["training"]["learning_rate"] = 0.006
            source.write_text(json.dumps(protocol))
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                run_protocol(source, "bad-reuse", output_root=root/"runs",
                             prepared_root=root/"prepared")
            # A positive public encoder needs its own matched controls.
            protocol.pop("reuse_controls_from")
            protocol["training"]["learning_rate"] = 0.0001
            protocol["stable_slots"]["trainable_mode"] = "positive_head"
            protocol["stable_slots"]["appearance_limits_by_method"]["non_private"] = [1, 1, 1]
            source.write_text(json.dumps(protocol))
            with patch("pdp.run.train_non_private", side_effect=AssertionError("wrong encoder control")):
                with contextlib.redirect_stdout(io.StringIO()):
                    head_result = run_protocol(source, "head", output_root=root/"runs",
                                               prepared_root=root/"prepared")
            head_manifest = json.loads((head_result/"manifest.json").read_text())
            self.assertTrue(all(c["status"] == "completed" for c in head_manifest["cells"]))
            initial = json.loads((head_result/"trusted/models/email_public_init_seed0/certificate.json").read_text())
            self.assertEqual(initial["schema"], "public-positive-head-initialization-v1")
            private = json.loads((head_result/"trusted/models/email_pdp_slots_seed0/certificate.json").read_text())
            self.assertEqual(private["private_parameter_count"], 5)
            head_manifest["cells"][-1]["status"] = "running"
            (head_result/"manifest.json").write_text(json.dumps(head_manifest))
            with patch("pdp.run.train_slots", side_effect=AssertionError("no repeated private call")):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_protocol(source, "head", resume=True, output_root=root/"runs",
                                 prepared_root=root/"prepared")

    def test_stable_slot_preregistration_rejects_invalid_layout_and_uniform_full_budget(self):
        protocol = json.loads((ROOT/"configs/pdp/stable-slots-pilot-v2.json").read_text())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"protocol.json"
            invalid = copy.deepcopy(protocol)
            invalid["stable_slots"]["bins_per_round"] = 3
            path.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                load_protocol(path)
            unfair = copy.deepcopy(protocol)
            unfair["base_epsilon"]["uniform_full"] = "4"
            path.write_text(json.dumps(unfair))
            with self.assertRaises(ValueError):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()

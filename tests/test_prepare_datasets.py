import tempfile
import unittest
from pathlib import Path

import json

from dataset.prepare_datasets import (
    DATASETS,
    DatasetSpec,
    prepare_dataset,
    prepare_paper_artifact,
)


class PrepareDatasetTests(unittest.TestCase):
    def test_fixed_facebook_paper_artifact_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = prepare_paper_artifact(
                DATASETS["facebook"], root, remove_self_loops=False, force=False
            )
            metadata = json.loads(
                (root / "facebook" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(stats["train_edges"], 43322)
            self.assertEqual(stats["test_edges"], 42089)
            self.assertEqual(metadata["split_source"], "paper-artifact")
            self.assertEqual(metadata["isolated_policy"], "artifact_edge_list")

    def test_isolated_split_members_are_retained_in_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "edges.txt").write_text("0 1\n1 2\n2 3\n", encoding="utf-8")
            spec = DatasetSpec("toy", "edges.txt", "whitespace", True)
            stats = prepare_dataset(
                spec, raw, root / "prepared", 0.5, 2, False, False, False
            )
            prepared = root / "prepared" / "toy"
            self.assertTrue((prepared / "metadata.json").is_file())
            self.assertTrue((prepared / "node_mapping.csv").is_file())
            self.assertEqual(
                stats["train_assigned_nodes"] + stats["test_assigned_nodes"], 4
            )

    def test_active_policy_drops_split_isolates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            (raw / "edges.txt").write_text("0 1\n1 2\n2 3\n", encoding="utf-8")
            spec = DatasetSpec("toy", "edges.txt", "whitespace", True)
            stats = prepare_dataset(
                spec, raw, root / "prepared", 0.5, 2, False, False,
                False, isolated_policy="active",
            )
            self.assertLessEqual(
                stats["train_active_nodes"] + stats["test_active_nodes"], 4
            )
            self.assertTrue(
                stats["train_active_nodes"] < stats["train_assigned_nodes"]
                or stats["test_active_nodes"] < stats["test_assigned_nodes"]
            )


if __name__ == "__main__":
    unittest.main()

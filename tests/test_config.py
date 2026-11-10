import unittest
import sys
from pathlib import Path
from unittest.mock import patch

from experiments.config import select_indicator_parameters


class IndicatorParameterTests(unittest.TestCase):
    def test_thesis_defaults_are_the_five_retained_datasets(self):
        from experiments.config import DATASETS
        expected = ('email', 'bitcoin', 'lastfm', 'hepph', 'facebook')
        self.assertEqual(DATASETS, expected)
        with patch.object(sys, 'path', [str(Path(__file__).resolve().parents[1] / 'experiments'), *sys.path]):
            from experiments.igraph_loader import PAPER_DATASETS
        from experiments.validate_results import DATASETS as VALIDATED
        from dataset.prepare_datasets import DATASETS as PREPARABLE
        self.assertEqual(PAPER_DATASETS, expected)
        self.assertEqual(VALIDATED, expected)
        self.assertNotIn('gowalla', PREPARABLE)

    def test_paper_indicator_uses_original_dataset_sizes(self):
        expected = {
            "email": (1005, (40, 4)),
            "bitcoin": (5881, (60, 4)),
            "lastfm": (7624, (60, 4)),
            "hepph": (12008, (60, 4)),
            "facebook": (22470, (60, 4)),
        }
        for dataset, (vertex_count, parameters) in expected.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    select_indicator_parameters(dataset, vertex_count), parameters
                )


if __name__ == "__main__":
    unittest.main()

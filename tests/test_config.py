import unittest

from experiments.config import select_indicator_parameters


class IndicatorParameterTests(unittest.TestCase):
    def test_paper_indicator_uses_original_dataset_sizes(self):
        expected = {
            "email": (1005, (40, 4)),
            "bitcoin": (5881, (60, 4)),
            "lastfm": (7624, (60, 4)),
            "hepph": (12008, (60, 4)),
            "facebook": (22470, (60, 4)),
            "gowalla": (196591, (80, 2)),
        }
        for dataset, (vertex_count, parameters) in expected.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    select_indicator_parameters(dataset, vertex_count), parameters
                )


if __name__ == "__main__":
    unittest.main()

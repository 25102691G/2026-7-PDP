import tempfile
import unittest
from pathlib import Path
import sys

import torch
import igraph
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from experiments.evaluate_models import load_net
from experiments.util import get_model
from experiments.graph_util import gen_deg_feature, gen_deg_feature_padded


class GratVariantTests(unittest.TestCase):
    def test_degree_feature_protocol_is_normalized_and_three_dimensional(self):
        graph = igraph.Graph(
            n=3, edges=[(0, 1), (0, 2), (1, 2)], directed=True
        )
        features = gen_deg_feature(graph)
        self.assertEqual(tuple(features.shape), (3, 3))
        self.assertTrue(torch.all(features[:, 0] == 1))
        self.assertGreaterEqual(float(features[:, 1:].min()), 0.0)
        self.assertLessEqual(float(features[:, 1:].max()), 1.0)

    def test_degree32_preserves_released_input_width(self):
        graph = igraph.Graph(
            n=3, edges=[(0, 1), (0, 2), (1, 2)], directed=True
        )
        features = gen_deg_feature_padded(graph, 32)
        self.assertEqual(tuple(features.shape), (3, 32))
        self.assertTrue(torch.all(features[:, 0] == 1))
        self.assertTrue(torch.all(features[:, 3:] == 0))

    def test_degree32_and_ones_keep_identical_model_initialization(self):
        torch.manual_seed(17)
        ones_model = get_model(
            "GRAT3", 32, *([32] * 5), grat_variant="legacy_released"
        )
        torch.manual_seed(17)
        degree_model = get_model(
            "GRAT3", 32, *([32] * 5), grat_variant="legacy_released"
        )
        for name, tensor in ones_model.state_dict().items():
            self.assertTrue(torch.equal(
                tensor, degree_model.state_dict()[name]
            ), name)

    def test_paper_grat_matches_appendix_parameterization(self):
        model = get_model(
            "GRAT3", *([32] * 6), grat_variant="paper_appendix_g"
        )
        first = model.grat.grat1
        self.assertIsNone(first.fc.bias)
        self.assertEqual(first.attn_fc.in_features, 64)

    def test_legacy_leaky_variant_changes_only_attention_activation(self):
        legacy = get_model(
            "GRAT3", *([32] * 6), grat_variant="legacy_released"
        ).grat.grat1
        leaky = get_model(
            "GRAT3", *([32] * 6), grat_variant="legacy_leaky_attention"
        ).grat.grat1
        self.assertEqual(legacy.fc.bias.shape, leaky.fc.bias.shape)
        self.assertEqual(
            legacy.attn_fc.weight.shape, leaky.attn_fc.weight.shape
        )
        with torch.no_grad():
            legacy.attn_fc.weight.zero_()
            legacy.attn_fc.bias.fill_(-1.0)
            leaky.attn_fc.weight.zero_()
            leaky.attn_fc.bias.fill_(-1.0)
        features = torch.ones(2, 32)
        edges = SimpleNamespace(
            src={"h": features}, dst={"h": features}
        )
        self.assertTrue(torch.equal(
            legacy.edge_attention(edges)["e"], torch.zeros(2, 1)
        ))
        self.assertTrue(torch.allclose(
            leaky.edge_attention(edges)["e"],
            torch.full((2, 1), -0.2),
        ))

    def test_loader_infers_legacy_checkpoint_without_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_GRAT3_d_1_seed_0_round_0.pt"
            legacy = get_model(
                "GRAT3", *([32] * 6), grat_variant="legacy_released"
            )
            torch.save(legacy.state_dict(), path)
            loaded = load_net(path, use_cuda=False)
            self.assertIsNotNone(loaded.grat.grat1.fc.bias)
            self.assertEqual(loaded.grat.grat1.attn_fc.in_features, 64)

    def test_loader_uses_checkpoint_feature_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_GRAT3_d_1_seed_0_round_0.pt"
            degree_model = get_model(
                "GRAT3", 3, *([32] * 5), grat_variant="legacy_released"
            )
            torch.save(degree_model.state_dict(), path)
            path.with_suffix(".json").write_text(
                '{"model_variant":"legacy_released",'
                '"feature_protocol":"degree","input_dim":3}',
                encoding="utf-8",
            )
            loaded = load_net(path, use_cuda=False)
            self.assertEqual(loaded.grat.grat1.fc.in_features, 3)

    def test_structural_initialization_preserves_positive_signal(self):
        torch.manual_seed(7)
        model = get_model(
            "GRAT3", *([32] * 6), grat_variant="legacy_released",
            grat_initialization="structural_preserving",
        )
        for layer in (model.grat.grat1, model.grat.grat2, model.grat.grat3):
            self.assertTrue(torch.all(layer.fc.weight >= 0))
            self.assertTrue(torch.allclose(
                layer.fc.weight.sum(dim=1),
                torch.ones(layer.fc.out_features),
                atol=1e-6,
            ))
            self.assertTrue(torch.all(layer.fc.bias == 0))
            self.assertTrue(torch.all(layer.attn_fc.bias == 0.1))

    def test_structural_initialization_remains_random(self):
        torch.manual_seed(7)
        first = get_model(
            "GRAT3", *([32] * 6), grat_variant="legacy_released",
            grat_initialization="structural_preserving",
        )
        torch.manual_seed(8)
        second = get_model(
            "GRAT3", *([32] * 6), grat_variant="legacy_released",
            grat_initialization="structural_preserving",
        )
        self.assertFalse(torch.equal(
            first.grat.grat1.fc.weight, second.grat.grat1.fc.weight
        ))


if __name__ == "__main__":
    unittest.main()

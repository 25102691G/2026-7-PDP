"""End-to-end contract checks on tiny graphs, never a privacy proof by testing."""

import contextlib
from dataclasses import replace
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch

os.environ.setdefault("DGLBACKEND", "pytorch")

import igraph
import torch

from pdp.accounting import RangePrivacy
from pdp.lattice import quantize_gradient
from pdp.training import (
    TrainingConfig, build_samples, initialize_model, save_release, train_b0,
    _contribution,
)


class B0TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self):
        return TrainingConfig(
            privacy=RangePrivacy(steps=3, batch_size=4, grid_bits=8),
            hidden_dim=4, subgraph_size=2, frequency_threshold=2,
            walk_length=10,
        )

    def graph(self):
        return igraph.Graph.Full(5, directed=True, loops=False)

    def test_tiny_graph_really_computes_nonzero_bounded_gradients(self):
        config = self.config()
        samples = build_samples(self.graph(), config)
        self.assertGreater(len(samples), 0)
        model = initialize_model(config)
        vector = _contribution(model, samples[0], config)
        self.assertTrue(any(vector))
        self.assertLessEqual(sum(x*x for x in vector), config.privacy.radius**2)
        # Existing Equation (5) loss and per-subgraph autograd remain the
        # source of the signal, rather than an unrelated toy optimizer.

    def test_exact_T_noise_calls_for_empty_sparse_and_normal_inputs(self):
        config = self.config()
        shapes = None
        for graph in [igraph.Graph(n=0, directed=True),
                      igraph.Graph(n=1, directed=True), self.graph()]:
            diagnostics = {}
            with patch("pdp.training.DiscreteGaussian.sample",
                       side_effect=lambda dimension: [1]*dimension) as noise:
                release = train_b0(graph, config, trusted_diagnostics=diagnostics)
            self.assertEqual(noise.call_count, config.privacy.steps)
            self.assertEqual(diagnostics["completed_steps"], config.privacy.steps)
            self.assertEqual(diagnostics["invalid_contributions"], 0, diagnostics)
            self.assertFalse(diagnostics["preprocessing_failed"], diagnostics)
            current = {key: tuple(value.shape) for key, value in release.parameters.items()}
            if shapes is None:
                shapes = current
            self.assertEqual(current, shapes)
            self.assertTrue(all(bool(torch.isfinite(x).all())
                                for x in release.parameters.values()))

    def test_failed_preprocessing_still_runs_same_noise_only_mechanism(self):
        config = self.config()
        with patch("pdp.training.DiscreteGaussian.sample",
                   side_effect=lambda dimension: [1]*dimension):
            empty = train_b0(igraph.Graph(n=0), config)
            trusted = {}
            with patch("pdp.training.build_samples", side_effect=RuntimeError("private")):
                failed = train_b0(self.graph(), config, trusted_diagnostics=trusted)
        self.assertTrue(trusted["preprocessing_failed"])
        self.assertEqual(empty.certificate, failed.certificate)
        for key in empty.parameters:
            self.assertTrue(torch.equal(empty.parameters[key], failed.parameters[key]))

    def test_isolated_graph_does_not_emit_structure_dependent_library_warnings(self):
        config = self.config()
        output = io.StringIO()
        with contextlib.redirect_stderr(output), contextlib.redirect_stdout(output):
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                train_b0(igraph.Graph(n=1), config)
        self.assertEqual(observed, [])
        self.assertEqual(output.getvalue(), "")

    def test_failed_and_nonfinite_gradients_are_not_released(self):
        config = self.config()
        trusted = {}
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            with patch("pdp.training._contribution", side_effect=RuntimeError("sensitive")):
                result = train_b0(self.graph(), config, trusted_diagnostics=trusted)
        self.assertGreater(trusted["invalid_contributions"], 0)
        self.assertEqual(output.getvalue(), "")
        encoded = json.dumps(result.certificate)
        self.assertNotIn("sensitive", encoded)
        self.assertNotIn("subgraph_count", encoded)
        self.assertNotIn("invalid_contributions", encoded)
        self.assertEqual(
            quantize_gradient([float("nan"), 1], radius=20, grid_bits=4), [0, 0],
        )

    def test_noise_is_added_before_float_conversion(self):
        config = replace(self.config(), privacy=RangePrivacy(
            steps=1, batch_size=4, grid_bits=8,
        ))
        radius = config.privacy.radius
        seen = []
        with patch("pdp.training._contribution", side_effect=lambda *args: (
            [radius] + [0]*(sum(p.numel() for p in args[0].parameters())-1)
        )):
            with patch("pdp.training.DiscreteGaussian.sample",
                       side_effect=lambda dimension: [10**100]*dimension):
                with patch("pdp.training._apply_noisy_integer_update",
                           side_effect=lambda model, vector, configuration: seen.append(vector)):
                    train_b0(self.graph(), config)
        self.assertEqual(len(seen), 1)
        self.assertTrue(all(isinstance(x, int) for x in seen[0]))
        self.assertEqual(seen[0][0], 10**100+4*radius)

    def test_safe_release_has_only_parameters_and_public_certificate(self):
        config = self.config()
        with patch("pdp.training.DiscreteGaussian.sample",
                   side_effect=lambda dimension: [1]*dimension):
            trusted = {}
            release = train_b0(self.graph(), config, trusted_diagnostics=trusted)
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)/"release"
            save_release(release, directory)
            self.assertEqual(
                sorted(path.name for path in directory.iterdir()),
                ["certificate.json", "parameters.pt"],
            )
            certificate = json.loads((directory/"certificate.json").read_text())
            self.assertEqual(certificate, release.certificate)
            tensors = torch.load(directory/"parameters.pt", weights_only=True)
            self.assertEqual(tensors.keys(), release.parameters.keys())
            with self.assertRaises(FileExistsError):
                save_release(release, directory)
        self.assertNotIn("subgraph_count", release.certificate)

    def test_production_noise_is_fresh_without_a_seed_interface(self):
        config = replace(self.config(), privacy=RangePrivacy(
            steps=1, batch_size=4, grid_bits=8,
        ))
        first = train_b0(igraph.Graph(n=0), config)
        second = train_b0(igraph.Graph(n=0), config)
        self.assertEqual(first.certificate, second.certificate)
        self.assertTrue(any(
            not torch.equal(first.parameters[key], second.parameters[key])
            for key in first.parameters
        ))


if __name__ == "__main__":
    unittest.main()

"""Synthetic training-aggregation checks; no real GNN fits in tests."""
import importlib.util
import random
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


class StableSlotTrainingTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.stable_slot_training'))
        from experiments import stable_slot_training
        return stable_slot_training

    def test_joint_clip_and_double_formula_all_2340_coordinates(self):
        api = self.api()
        templates = [torch.zeros(1000), torch.zeros(1340)]
        a = torch.zeros(4, 4, 2340)
        for j in range(4):
            for i in range(4):
                raw = torch.zeros(2340)
                raw[j] = 3.; raw[1000 + j] = 4.
                clipped, info, _ = api.safe_clip([raw[:1000], raw[1000:]], templates)
                self.assertAlmostEqual(info['legacy_pre_norm'], 5.)
                self.assertLessEqual(info['post_norm'], 3.00001)
                a[j, i] = clipped
        result = api.gd.aggregate(a, templates)
        u = a.double().sum(1)
        coefs = torch.tensor([min(1., 3. / (float(v.norm()) + 1e-6)) for v in u], dtype=torch.float64)
        expected = (u * coefs[:, None]).sum(0)
        self.assertTrue(torch.allclose(result['slot'].double(), expected, atol=2e-6))
        self.assertTrue(torch.allclose(result['flat'].double(), a.double().sum((0, 1)), atol=2e-6))
        self.assertTrue(all(api.gd.norm(v) <= 3.00001 for v in result['v']))
        # Whole batch is deliberately NOT clipped to C=3.
        self.assertGreater(api.gd.norm(result['slot']), 3.)

    def test_update_once_uses_sum_divided_by_16_without_final_clip_or_noise(self):
        api = self.api()
        p = torch.nn.Parameter(torch.ones(1000)); q = torch.nn.Parameter(torch.ones(1340))
        signal = torch.arange(2340, dtype=torch.float32) / 100
        before = torch.cat([p.detach(), q.detach()]).double()
        audit = api.apply_update([p, q], signal)
        after = torch.cat([p.detach(), q.detach()]).double()
        self.assertTrue(torch.allclose(after, before - .005 * signal.double() / 16, atol=2e-7))
        self.assertEqual(audit['updates'], 1)
        self.assertEqual(audit['denominator'], 16)
        self.assertEqual(audit['coordinates'], 2340)
        self.assertTrue(audit['changed'])

    def test_empty_short_and_nonfinite_inputs_have_fixed_zero_positions(self):
        api = self.api()
        for sizes in ([0, 1, 2, 3], [10, 0, 7, 4]):
            schedule, _ = api.make_schedule(sizes, iterations=3)
            self.assertEqual(len(schedule), 3)
            for step in schedule:
                self.assertEqual([len(x) for x in step], [4] * 4)
                for n, indices in zip(sizes, step):
                    kept = [i for i in indices if i is not None]
                    self.assertEqual(len(kept), min(n, 4))
                    self.assertEqual(len(set(kept)), len(kept))
        clipped, info, raw = api.safe_clip([torch.tensor([float('inf')]), None], [torch.zeros(1), torch.zeros(2)])
        self.assertEqual(api.gd.norm(clipped), 0.)
        self.assertEqual(info['nonfinite_coordinates'], 1)
        self.assertTrue(info['zero_substituted'])

    def test_schedule_matches_diagnostic_first_batch_and_is_rng_isolated(self):
        api = self.api()
        state = random.getstate()
        schedule, streams = api.make_schedule([16, 13, 16, 14])
        self.assertEqual(len(schedule), 100)
        self.assertEqual(schedule, api.make_schedule([16, 13, 16, 14])[0])
        for j, n in enumerate([16, 13, 16, 14]):
            self.assertEqual(schedule[0][j], api.gd.choose_indices(n, 0, j)[0])
        self.assertEqual(state, random.getstate())
        self.assertEqual(len({s['seed'] for s in streams}), 4)

    def test_local_stream_not_affected_by_other_slot_container_size(self):
        api = self.api()
        a, ra = api.make_schedule([16, 13, 16, 14])
        b, rb = api.make_schedule([0, 13, 16, 14])
        for j in (1, 2, 3):
            self.assertEqual([r[j] for r in a], [r[j] for r in b])
            self.assertEqual(ra[j], rb[j])

    def test_smoke_gate_dispatches_only_twelve_authorized_cases(self):
        api = self.api()
        order = api.case_order()
        self.assertEqual(len(order), 12)
        self.assertEqual(len(set(order)), 12)
        self.assertTrue(all(seed == 0 for seed, arm, branch in order[:4]))
        calls = []
        def fit(case):
            calls.append(case)
            return dict(passed=False)
        with self.assertRaises(ValueError):
            api.dispatch(fit, lambda rows: False)
        self.assertEqual(calls, order[:4])
        calls.clear()
        result = api.dispatch(lambda case: calls.append(case) or case, lambda rows: True)
        self.assertEqual(calls, order)
        self.assertEqual(len(result), 12)

    def test_training_guards_disallow_optimizer_and_noise(self):
        api = self.api()
        with api.training_guards():
            with self.assertRaises(AssertionError):
                torch.randn(3)
            with self.assertRaises(AssertionError):
                torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.005)

    def test_preflight_covers_layout_formula_and_no_final_batch_clip(self):
        result = self.api().synthetic_preflight()
        self.assertTrue(result['passed'])
        self.assertEqual(result['coordinates'], 2340)
        self.assertEqual(result['updates_per_step'], 1)
        self.assertGreater(result['double_batch_norm'], 3.)

    def test_diagnostic_csv_writer_is_available(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'steps.csv'
            api.write_csv(target, [dict(step=100, updates=1)])
            self.assertIn('100,1', target.read_text())

    def test_loss_reset_and_zero_padding_do_not_reuse_previous_gradient(self):
        api = self.api()
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(1.)
        # Synthetic scalar loss, no GNN or private graph.
        point = dict(x=torch.tensor([[2.]]))
        def loss(m, item): return m(item['x']).sum()
        batch = [[point, None, None, None]] + [[None] * 4 for _ in range(3)]
        result = api.batch_gradients(model, batch, loss)
        self.assertEqual(len(result['samples']), 16)
        self.assertEqual(result['a'][0, 0, 0].item(), 2.)
        self.assertEqual(torch.count_nonzero(result['a']).item(), 1)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()

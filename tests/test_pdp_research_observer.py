"""Artificial-only checks that diagnostic observation leaves the algorithm intact."""
import importlib.util
import sys
import unittest
from unittest.mock import patch

import torch


class ObserverTests(unittest.TestCase):
    def observer(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.pdp_research_observer'))
        from experiments.pdp_research_observer import TrainingObserver
        return TrainingObserver

    def test_cache_observation_preserves_members_frequencies_and_random_states(self):
        Observer = self.observer()
        from experiments.node_dp_slots.adapter import SlotCache
        ids = tuple(range(96))
        edges = [(v, (v + k) % 96) for v in ids for k in range(1, 13)]
        plain = SlotCache(ids, edges, True, 0)
        with Observer() as observer:
            observed = SlotCache(ids, edges, True, 0)
        self.assertEqual(plain.fingerprint(), observed.fingerprint())
        slot = observer.summary()['sampling_slots'][0]
        self.assertEqual(slot['candidates'], len(plain.subgraphs))
        self.assertEqual(slot['scs_candidates'] + slot['bes_candidates'], slot['candidates'])
        self.assertEqual(slot['rwr_successes'], slot['candidates'])
        self.assertGreaterEqual(slot['rwr_attempts'], slot['rwr_successes'])

    def test_observed_training_matches_unobserved_parameters_and_diagnostics(self):
        Observer = self.observer()
        from tests.test_pdp_research import ResearchTests
        from experiments.node_pdp_research import Session
        torch.set_num_threads(1)
        registry, spec, graph = ResearchTests().fixture()
        plain = Session.synthetic(registry, spec, 'node_pdp_v2_projected', sampling_seed=17, noise_seed=19).run(graph)
        session = Session.synthetic(registry, spec, 'node_pdp_v2_projected', sampling_seed=17, noise_seed=19)
        with Observer() as observer:
            observed = session.run(graph)
        self.assertEqual(plain.parameters, observed.parameters)
        self.assertEqual(plain.diagnostics, observed.diagnostics)
        result = observer.summary()
        self.assertEqual(len(result['steps']), 100)
        self.assertEqual(len(result['sampling_slots']), 4)
        for frame in result['steps']:
            self.assertEqual(sum(s['effective_positions'] + s['zero_padding_positions'] for s in frame['slots']), 16)
            for slot in frame['slots']:
                self.assertEqual(slot['selected_scs_positions'] + slot['selected_bes_positions'], slot['effective_positions'])
                self.assertIsNotNone(slot['slot_preclip_norm'])
                self.assertLessEqual(slot['slot_postclip_norm'], 3.)

    def test_empty_slots_record_padding_and_zero_slot_norms(self):
        Observer = self.observer()
        from tests.test_pdp_research import ResearchTests
        from experiments.node_pdp_research import Session
        from experiments.node_dp_slots.adapter import Graph
        registry, spec, _ = ResearchTests().fixture()
        with Observer() as observer:
            Session.synthetic(registry, spec, 'node_pdp_v2_projected', sampling_seed=0, noise_seed=0).run(Graph((), (), True))
        result = observer.summary()
        self.assertEqual([s['candidates'] for s in result['sampling_slots']], [0] * 4)
        self.assertEqual(sum(s['zero_padding_positions'] for f in result['steps'] for s in f['slots']), 1600)
        self.assertTrue(all(s['slot_postclip_norm'] == 0. for f in result['steps'] for s in f['slots']))

    def test_resource_error_not_masked_and_profiler_removed(self):
        Observer = self.observer()
        from experiments.node_dp_slots import adapter
        with patch.object(adapter, 'sample_privim_star', side_effect=MemoryError('artificial-resource-failure')):
            with self.assertRaisesRegex(MemoryError, 'artificial-resource-failure'):
                with Observer():
                    adapter.SlotCache((), (), True, 0)
        self.assertIsNone(sys.getprofile())


if __name__ == '__main__':
    unittest.main()

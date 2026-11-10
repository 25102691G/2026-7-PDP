"""Scope/aggregation/reconstruction tests; no real GNN training."""
import importlib.util
import statistics
import unittest
from unittest.mock import patch


class RepeatTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.email_h2_repeat.run'))
        from experiments.email_h2_repeat import run
        return run

    def test_scope_is_exactly_six_and_excludes_seed0(self):
        r=self.api()
        self.assertEqual(r.CASES,tuple((seed,arm) for seed in (1,2) for arm in r.u.ARMS))
        r.validate_scope(r.CONFIG,r.CASES)
        for cases in (r.CASES+((0,'original'),),r.CASES[:-1],r.CASES[::-1]):
            with self.assertRaises(ValueError):r.validate_scope(r.CONFIG,cases)
        for key,value in [('seeds',[0,1,2]),('noise_level','H1'),('gaussian_std_on_sum',100.)]:
            bad=dict(r.CONFIG);bad[key]=value
            with self.assertRaises(ValueError):r.validate_scope(bad,r.CASES)

    def test_dispatch_passes_exactly_six_or_none(self):
        r=self.api()
        with patch.object(r.h,'synthetic_preflight',side_effect=ValueError('failed')):
            with patch.object(r,'train_case') as train:
                with self.assertRaises(ValueError):r.dispatch(lambda case:case)
                train.assert_not_called()
        with patch.object(r.h,'synthetic_preflight',return_value={'passed':True}):
            with patch.object(r,'train_case',side_effect=lambda case:case) as train:
                rows=r.dispatch(lambda case:case)
                self.assertEqual(rows,list(r.CASES));self.assertEqual(train.call_count,6)

    def test_seed0_is_read_only_metadata_not_recomputed(self):
        r=self.api()
        with patch.object(r.u,'new_model',side_effect=AssertionError('seed0 must not be inferred/retrained')):
            rows,dominance=r.read_seed0()
        self.assertEqual({(x['seed'],x['arm']) for x in rows},{(0,a) for a in r.u.ARMS})
        self.assertEqual(len(dominance),3)
        original=next(x for x in rows if x['arm']=='original')
        self.assertAlmostEqual(original['h2_coverage'],98.14814814814815)

    def test_paired_summary_uses_within_seed_differences(self):
        r=self.api();rows=[]
        for seed,(h2,noise) in enumerate(((91.,90.),(93.,90.),(89.,90.))):
            rows.append(dict(seed=seed,arm='original',h2_coverage=h2,noise_only_coverage=noise,
                             h2_minus_noise_only_pp=h2-noise))
        summary=r.summarize(rows,metrics=('h2_coverage','noise_only_coverage','h2_minus_noise_only_pp'))[0]
        self.assertEqual(summary['runs'],3)
        self.assertEqual(summary['h2_minus_noise_only_pp_mean'],1.)
        self.assertEqual(summary['h2_minus_noise_only_pp_std'],statistics.stdev([1.,3.,-1.]))

    def test_historical_seed0_metric_mismatch_stops_merge(self):
        r=self.api();original=r.u.read_csv
        def modified(path):
            rows=original(path)
            if path==r.H2_ZERO/'results.csv':
                rows=[dict(row) for row in rows];rows[0]['influence_spread']='1'
            return rows
        with patch.object(r.u,'read_csv',side_effect=modified):
            with self.assertRaises(ValueError):r.read_seed0()

    def test_noise_only_rebuild_accepts_no_gradient_input(self):
        r=self.api();t=r.torch
        with r.offline.inference_guards():
            result=r.rebuild_noise_only([t.tensor([1.])],[[t.tensor([16.])]])
        self.assertTrue(t.equal(result['closed32'][0],t.tensor([.995])))

    def test_noise_seed_collision_aborts_without_redraw(self):
        r=self.api()
        with patch.object(r.secrets,'randbits',side_effect=[1,2,3,4,5,6]) as draw:
            with self.assertRaises(ValueError):r.make_noise_seeds({1})
            self.assertEqual(draw.call_count,6)


if __name__=='__main__':unittest.main()

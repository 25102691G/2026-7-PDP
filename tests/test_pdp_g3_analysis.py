"""Pure artificial outcome tests; no torch/model runtime."""
import unittest


class G3AnalysisTests(unittest.TestCase):
    def test_paired_statistics_and_negative_cases_preserved(self):
        from experiments.pdp_g3_analysis import summary
        s=summary([90,90,90,90,40],[95,95,95,70,55])
        self.assertEqual(s['deltas'],[5,5,5,-20,15])
        self.assertEqual(s['maximum_decline_pp'],20)
        self.assertEqual(s['high_last_drop_ge5_count'],1)
        self.assertEqual(s['low_last_count'],1);self.assertEqual(s['low_avg_count'],0)
        self.assertFalse(s['screen_pass'])
        self.assertFalse(s['checks']['no_high_last_drop_ge5'])

    def test_positive_constant_gain_can_pass_all_fixed_checks(self):
        from experiments.pdp_g3_analysis import summary
        s=summary([20,40,60,80,100],[25,45,65,85,105])
        self.assertEqual(s['delta_mean'],5)
        self.assertEqual(s['delta_sample_sd'],0)
        self.assertTrue(s['screen_pass'])


if __name__=='__main__':unittest.main()

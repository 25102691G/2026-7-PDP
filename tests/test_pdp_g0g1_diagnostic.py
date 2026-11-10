"""Artificial set algebra and resource counters only; no GNN forwards."""
import unittest


class DiagnosticTests(unittest.TestCase):
    def api(self):
        from experiments import pdp_g0g1_diagnostic
        return pdp_g0g1_diagnostic

    def test_directed_reach_and_redundancy(self):
        api=self.api();neighborhoods=api.closed_neighborhoods(5,[(0,1),(0,2),(1,2),(3,4)],True)
        r=api.coverage_stats(neighborhoods,[0,1])
        self.assertEqual((r['spread'],r['A']), (3,5))
        self.assertAlmostEqual(r['R'],.4)
        self.assertEqual(r['ordered_marginal_gains'],[3,0])
        self.assertEqual(r['leave_one_out_unique'],[1,0])

    def test_undirected_and_isolated_nodes(self):
        api=self.api();n=api.closed_neighborhoods(4,[(0,1),(1,2)],False)
        self.assertEqual(n[3],{3})
        r=api.coverage_stats(n,[0,2]);self.assertEqual((r['spread'],r['A']), (3,4))
        self.assertAlmostEqual(r['R'],.25)

    def test_symmetric_difference_decomposition_is_exact(self):
        api=self.api();r=api.decompose(5,.4,8,.25)
        self.assertAlmostEqual(r['reach_term']+r['redundancy_term'],6-3)

    def test_spearman_handles_ties_and_constant_scores(self):
        api=self.api();self.assertAlmostEqual(api.spearman([1,1,2,3],[1,1,2,3]),1)
        self.assertAlmostEqual(api.spearman([1,1,2,3],[-1,-1,-2,-3]),-1)
        self.assertIsNone(api.spearman([1,1],[2,3]))

    def test_forward_limit_charges_attempts_before_execution(self):
        api=self.api();b=api.Budget(2,600,clock=lambda:0)
        with self.assertRaises(ValueError):b.call(lambda: (_ for _ in ()).throw(ValueError('artificial')))
        self.assertEqual(b.call(lambda:7),7)
        with self.assertRaises(RuntimeError):b.call(lambda:8)
        self.assertEqual((b.attempts,b.completed),(2,1))

    def test_elapsed_budget_blocks_without_calling_function(self):
        api=self.api();now=[0];b=api.Budget(30,600,clock=lambda:now[0]);now[0]=601
        with self.assertRaises(TimeoutError):b.call(lambda:None)
        self.assertEqual(b.attempts,0)


if __name__=='__main__':unittest.main()

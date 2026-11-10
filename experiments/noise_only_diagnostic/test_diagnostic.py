"""Offline reconstruction and inference guards; never train a GNN."""
import importlib.util
import random
import unittest


class DiagnosticTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.noise_only_diagnostic.run'))
        from experiments.noise_only_diagnostic import run
        return run

    def test_closed_form_and_fp32_reconstruction(self):
        r=self.api();t=r.torch
        initial=[t.tensor([1.,2.])]
        frames=[[t.tensor([10.,-3.])],[t.tensor([-4.,7.])]]
        result=r.reconstruct_noise_only(initial,frames)
        expected=initial[0].double()-(.005/16)*t.tensor([6.,4.]).double()
        self.assertTrue(t.equal(result['closed64'][0],expected))
        sequential=initial[0].clone()
        for frame in frames:sequential.add_(frame[0]/16,alpha=-.005)
        self.assertTrue(t.equal(result['sequential32'][0],sequential))
        self.assertFalse(result['closed64'][0].requires_grad)
        self.assertTrue(t.equal(initial[0],t.tensor([1.,2.])))

    def test_hidden_optimizer_operations_are_rejected(self):
        r=self.api();config=dict(learning_rate=.005,batch_size=16,momentum=0.,lr_decay=1.)
        r.assert_plain_sgd(config,r.sgd_defaults())
        for key,value in [('momentum',.9),('lr_decay',.99),('ema',True),('weight_decay',.01)]:
            bad=dict(config);bad[key]=value
            with self.assertRaises(ValueError):r.assert_plain_sgd(bad,r.sgd_defaults())

    def test_actual_step_witness_rejects_hidden_update(self):
        r=self.api();t=r.torch
        before=[t.tensor([1.])];signal=[t.tensor([2.])];noise=[t.tensor([3.])]
        used=[(signal[0]+noise[0])/16];after=[before[0].add(used[0],alpha=-.005)]
        packet=dict(before=before,signal_sum=signal,used_gradient=used,after=after)
        r.verify_saved_step(packet,noise)
        packet['after']=[after[0]+.01]
        with self.assertRaises(ValueError):r.verify_saved_step(packet,noise)

    def test_incoming_edges_do_not_count_as_outgoing_influence(self):
        r=self.api();import igraph
        graph=igraph.Graph(n=4,edges=[(0,1),(2,0),(2,3)],directed=True)
        metrics=r.selection_metrics(graph,[0])
        self.assertEqual(metrics['influence_spread'],2)
        self.assertEqual(metrics['mean_selected_outdegree'],1.)
        self.assertEqual(r.degree_topk(graph,2),[2,0])
        self.assertEqual(r.selection_metrics(graph,[1])['selected_zero_outdegree_count'],1)

    def test_random_reference_seeds_are_frozen_and_local(self):
        r=self.api();state=random.getstate()
        self.assertEqual(r.RANDOM_SEEDS,tuple(range(20)))
        a=r.random_sets(458);b=r.random_sets(458)
        self.assertEqual(a,b);self.assertEqual(len(a),20)
        self.assertTrue(all(len(s)==len(set(s))==50 for s in a.values()))
        self.assertEqual(state,random.getstate())

    def test_tie_and_constant_correlation_handling(self):
        r=self.api();scores=r.torch.ones(4)
        diag=r.logit_diagnostics(scores,[0,1,2,3],2)
        self.assertEqual(diag['unique_logits'],1)
        self.assertEqual(diag['boundary_tie_count'],4)
        self.assertIsNone(diag['logit_outdegree_spearman'])

    def test_gradient_and_optimizer_calls_are_blocked(self):
        r=self.api();t=r.torch
        with r.inference_guards():
            with self.assertRaises(RuntimeError):t.autograd.grad(t.tensor(1.),t.tensor(1.))
            with self.assertRaises(RuntimeError):t.tensor(1.,requires_grad=True).backward()
            with self.assertRaises(RuntimeError):t.optim.SGD.step(None)


if __name__=='__main__':unittest.main()

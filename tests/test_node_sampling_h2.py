"""Synthetic noise checks only; no GNN training in these tests."""
import importlib.util
import unittest
from unittest.mock import patch


class H2Tests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.node_sampling_h2'))
        from experiments import node_sampling_h2
        return node_sampling_h2

    def test_joint_clip_sum_noise_then_divide_and_exact_update(self):
        api = self.api(); torch = api.torch
        a = torch.nn.Parameter(torch.tensor([1.])); b = torch.nn.Parameter(torch.tensor([2.]))
        opt = api.FixedH2SGD([{'params':[a]}, {'params':[b]}], noise_seed=123)
        opt.zero_accum_grad()
        for _ in range(16):
            opt.zero_microbatch_grad(); a.grad=torch.tensor([3.]); b.grad=torch.tensor([4.])
            opt.microbatch_step()
        with patch.object(torch, 'randn', side_effect=lambda shape, **kw: torch.full(shape, 2., dtype=kw['dtype'], device=kw['device'])):
            opt.step()
        coefficient=3/(5+1e-6)
        expected=torch.tensor([1.,2.])-.005*(torch.tensor([3.,4.])*coefficient+2*6.48193359375)
        self.assertTrue(torch.allclose(torch.cat([a.detach(),b.detach()]), expected))
        record=opt.records[-1]
        self.assertTrue(record['update_verified'] and record['noise_verified'])
        self.assertEqual(record['noise_coordinate_count'], 2)
        self.assertEqual(record['clip_fraction'], 1.)
        self.assertEqual(opt.l2_norm_clip*opt.noise_multiplier,103.7109375)

    def test_noise_covers_all_model_coordinates(self):
        api=self.api();torch=api.torch
        params=api.synthetic_parameters()
        self.assertEqual(sum(p.numel() for p in params),2340)
        opt=api.FixedH2SGD(params,noise_seed=234)
        opt.zero_accum_grad()
        for _ in range(16):
            opt.zero_microbatch_grad()
            for p in params:p.grad=torch.zeros_like(p)
            opt.microbatch_step()
        opt.step()
        self.assertEqual(opt.records[-1]['noise_coordinate_count'],2340)
        self.assertTrue(all(bool((p.grad!=0).all()) for p in params))
        self.assertTrue(opt.records[-1]['global_rng_unchanged'])

    def test_multi_draw_statistics_and_stream_isolation(self):
        result=self.api().synthetic_preflight(draws=128)
        self.assertTrue(result['passed'])
        self.assertEqual(result['draws_per_stream'],128)
        self.assertEqual(result['coordinates'],2340)
        self.assertTrue(result['all_streams_distinct'])
        self.assertTrue(result['global_rng_unchanged'])

    def test_single_noise_vector_cannot_pass_preflight(self):
        with self.assertRaises(ValueError):self.api().synthetic_preflight(draws=1)

    def test_unsupported_scope_or_noise_overrides_are_rejected(self):
        api=self.api()
        for key,value in [('seeds',[1]),('noise_level','H1'),('gaussian_std_on_sum',1.),('iterations',101)]:
            config=dict(api.CONFIG);config[key]=value
            with self.assertRaises(ValueError):api.validate_scope(config)
        with self.assertRaises(TypeError):
            api.FixedH2SGD(api.synthetic_parameters(),noise_seed=0,noise_multiplier=1.)

    def test_failed_preflight_never_starts_training(self):
        api=self.api()
        with patch.object(api,'synthetic_preflight',side_effect=ValueError('bad variance')):
            with patch.object(api,'train_case') as train:
                with self.assertRaises(ValueError):api.run_approved_cases(lambda _:None)
                train.assert_not_called()

    def test_passed_gate_dispatches_exactly_three_approved_cases(self):
        api=self.api()
        with patch.object(api,'synthetic_preflight',return_value={'passed':True}):
            with patch.object(api,'train_case',side_effect=lambda context:context) as train:
                _,rows=api.run_approved_cases(lambda arm:dict(arm=arm,seed=0))
        self.assertEqual(train.call_count,3)
        self.assertEqual(rows,[dict(arm=arm,seed=0) for arm in ('original','personalized','uniform_matched')])

    def test_incomplete_batch_and_nonfinite_gradients_rejected(self):
        api=self.api();torch=api.torch
        p=torch.nn.Parameter(torch.zeros(1));opt=api.FixedH2SGD([p],noise_seed=5)
        opt.zero_accum_grad();p.grad=torch.full_like(p,float('nan'))
        with self.assertRaises(ValueError):opt.microbatch_step()
        with self.assertRaises(ValueError):opt.step()


if __name__=='__main__':unittest.main()

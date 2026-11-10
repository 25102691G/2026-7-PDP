"""Only artificial graphs/vectors; no prepared data or real-data training."""
import importlib.util
import json
import math
import random
import unittest
from unittest.mock import patch

import torch


class NodeDPSlotTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.node_dp_slots'))
        from experiments.node_dp_slots import config,adapter,aggregation,accounting,noise,mechanism,release
        return config,adapter,aggregation,accounting,noise,mechanism,release

    def fixture(self,directed=True):
        c,a,*_=self.api();spec=c.Spec(tuple(range(128)),directed=directed)
        edges=[(i,(i+k)%96) for i in range(96) for k in range(1,13)]
        if not directed:edges=sorted(set(tuple(sorted(e)) for e in edges))
        return spec,a.Graph(tuple(range(96)),tuple(edges),directed)

    def test_fixed_spec_universe_and_private_domain(self):
        c,a,*_=self.api();spec=c.Spec((1,2,3))
        self.assertEqual((spec.B,spec.C,spec.T,spec.s_sum),(16,3.,100,260.))
        with self.assertRaises(Exception):spec.B=4
        with self.assertRaises(ValueError):c.Spec((1,1))
        with self.assertRaises(ValueError):a.Adapter(spec,a.Graph((4,),(),True))
        self.assertEqual(c.memberships(123,0),c.memberships(123,0))
        for v in range(500):
            pair=c.memberships(v,0)
            self.assertIn(pair[0],(0,1));self.assertIn(pair[1],(2,3))

    def test_locality_caches_frequency_rng_and_local_reindexing(self):
        c,a,*_=self.api()
        for directed in (True,False):
            spec,graph=self.fixture(directed)
            before=a.Adapter(spec,graph)
            reordered=a.Adapter(spec,a.Graph(tuple(reversed(graph.nodes)),tuple(reversed(graph.edges)),directed))
            self.assertEqual(before.fingerprints(),reordered.fingerprints())
            for v in (0,47,95):
                after=a.Adapter(spec,graph.delete(v))
                untouched=set(range(4))-set(c.memberships(v,0))
                for j in untouched:
                    self.assertEqual(before.fingerprints()[j],after.fingerprints()[j])
                p=a.Adapter(spec,graph);q=a.Adapter(spec,graph.delete(v))
                for _ in range(3):
                    bp,bq=p.batch(),q.batch()
                    for j in untouched:
                        self.assertEqual(bp[j],bq[j]);self.assertEqual(p.caches[j].batch_rng.getstate(),q.caches[j].batch_rng.getstate())
            self.assertTrue(all(max(x.frequencies,default=0)<=4 for x in before.caches))

    def test_empty_isolated_short_and_partial_failure_padding(self):
        c,a,*_=self.api()
        for n in (0,1,8):
            adapted=a.Adapter(c.Spec(tuple(range(10))),a.Graph(tuple(range(n)),(),True))
            self.assertEqual([len(x) for x in adapted.batch()],[4]*4)
            self.assertTrue(all(item is None for slot in adapted.batch() for item in slot))
        self.assertEqual(a.choose_positions([10,20],random.Random(1)).count(None),2)
        spec,graph=self.fixture();adapted=a.Adapter(spec,graph)
        self.assertTrue(any(cache.subgraphs for cache in adapted.caches))
        self.assertTrue(all(len(x)==4 for x in adapted.batch()))

    def test_adapter_does_not_share_global_random_state(self):
        c,a,*_=self.api();spec,graph=self.fixture();state=random.getstate()
        adapted=a.Adapter(spec,graph);adapted.batch()
        self.assertEqual(state,random.getstate())
        self.assertEqual(len({id(x.sample_rng) for x in adapted.caches}),4)
        self.assertEqual(len({id(x.batch_rng) for x in adapted.caches}),4)
        self.assertEqual(len({id(x.frequencies) for x in adapted.caches}),4)

    def test_joint_double_clip_bound_and_nonfinite_totalization(self):
        _,_,agg,*_=self.api()
        clipped=agg.clip(torch.tensor([3.,4.]));self.assertLessEqual(float(clipped.norm()),3.)
        self.assertEqual(float(agg.clip(torch.tensor([float('nan'),1.])).norm()),0.)
        self.assertEqual(float(agg.clip(torch.tensor([float('inf'),1.])).norm()),0.)
        a=torch.zeros(4,4,2340,dtype=torch.float64)
        for j in range(4):a[j,:,j]=3
        s,b=agg.double_clip(a)
        self.assertTrue(all(float(x.norm())<=3 for x in b))
        self.assertGreater(float(s.norm()),3.)
        self.assertTrue(torch.allclose(s,b.sum(0)))
        # Adversarial replacement of two slot contributions reaches almost12.
        b1=torch.zeros(4,2340,dtype=torch.float64);b2=b1.clone()
        b1[:2,0]=3;b2[:2,0]=-3
        self.assertEqual(float((b1.sum(0)-b2.sum(0)).norm()),12.)

    def test_missing_gradients_cover_all_coordinates(self):
        _,_,agg,*_=self.api()
        params=[torch.nn.Parameter(torch.zeros(1000)),torch.nn.Parameter(torch.zeros(1340))]
        params[0].grad=torch.ones_like(params[0])
        v=agg.read_gradient(params)
        self.assertEqual(v.numel(),2340);self.assertEqual(float(v[1000:].norm()),0.)
        self.assertLessEqual(float(agg.clip(v).norm()),3.)

    def test_independent_rdp_accountant_and_composition(self):
        _,_,_,acc,*_=self.api()
        e,alpha=acc.epsilon()
        self.assertAlmostEqual(e,1.98209273899,places=9);self.assertEqual(alpha,10.3)
        self.assertAlmostEqual(acc.minimum_scale(),257.8932391005,places=6)
        self.assertGreater(acc.epsilon(models=2)[0],2.)
        ledger=acc.Ledger();ledger.charge()
        with self.assertRaises(ValueError):ledger.charge()
        self.assertEqual(ledger.models,1)

    def test_noise_modes_scaling_and_no_public_seed_api(self):
        *_,noise,mech,release=self.api()
        with self.assertRaises(TypeError):noise.SecretGaussian(seed=0)
        n=noise.TestGaussian(9);draw=n.draw()
        self.assertEqual(draw.shape,(2340,));self.assertEqual(n.calls,1)
        self.assertTrue(bool((draw!=0).all()))
        values=torch.cat([n.draw() for _ in range(96)])
        self.assertLess(abs(float(values.mean())),2.)
        self.assertLess(abs(float(values.std())-260),2.)
        with self.assertRaises(TypeError):mech.Engine.privacy(None,noise=noise.TestGaussian(1))

    def test_single_noise_then_fixed_denominator_once(self):
        *_,noise,mech,release=self.api()
        p=torch.nn.Parameter(torch.ones(1000));q=torch.nn.Parameter(torch.ones(1340))
        vector=torch.arange(2340,dtype=torch.float64)/1000
        class Fixed:
            calls=0
            def draw(self):self.calls+=1;return torch.full((2340,),260.,dtype=torch.float64)
        n=Fixed();old=torch.cat([p.detach(),q.detach()]).double()
        mech.update([p,q],vector,n)
        new=torch.cat([p.detach(),q.detach()]).double()
        self.assertTrue(torch.allclose(new,old-.005*(vector+260)/16,atol=1e-7))
        self.assertEqual(n.calls,1)

    def test_release_whitelist_test_mode_and_strict_gate(self):
        c,a,agg,acc,n,mech,r=self.api()
        spec=c.Spec(tuple(range(4)))
        test_run=mech.Engine.testing(spec,noise_seed=0).run(a.Graph((),(),True))
        self.assertEqual(test_run.testing_diagnostics['updates'],100)
        self.assertEqual(test_run.testing_diagnostics['noise_calls'],100)
        for ideal in (False,True):
            with self.assertRaises(PermissionError):r.export(test_run,permit_idealized=ideal)
        private=mech.Engine.privacy(spec).run(a.Graph((),(),True))
        with self.assertRaises(PermissionError):r.export(private)
        payload=r.export(private,permit_idealized=True)
        self.assertEqual(set(payload),{'parameters','configuration'})
        self.assertEqual(len(payload['parameters']),2340)
        self.assertFalse(payload['configuration']['machine_dp_certified'])
        fields=set(payload)|set(payload['configuration'])
        for key in ('noise_seed','loss','container_size','frequencies','private_ids','raw_gradient','noise_norm'):
            self.assertNotIn(key,fields)
        self.assertIsNone(private.testing_diagnostics)

    def test_synthetic_real_model_100_steps_and_adjacency_center(self):
        c,a,agg,acc,n,mech,r=self.api();spec,graph=self.fixture()
        engine=mech.Engine.testing(spec,noise_seed=5)
        state=mech.new_model()
        left=a.Adapter(spec,graph);right=a.Adapter(spec,graph.delete(47))
        p,bp=mech.center(state,left.batch());q,bq=mech.center(state,right.batch())
        self.assertLessEqual(float((p-q).norm()),12.)
        for j in set(range(4))-set(c.memberships(47,0)):
            self.assertTrue(torch.equal(bp[j],bq[j]))
        result=engine.run(graph)
        self.assertEqual(result.testing_diagnostics['updates'],100)
        self.assertEqual(result.testing_diagnostics['coordinates'],2340)
        self.assertTrue(result.testing_diagnostics['finite_parameters'])

    def test_nonfinite_model_postprocessing_and_resource_gap(self):
        c,a,agg,acc,n,mech,r=self.api()
        p=torch.nn.Parameter(torch.zeros(2340))
        class BadNoise:
            def draw(self):return torch.full((2340,),float('inf'),dtype=torch.float64)
        mech.update([p],torch.zeros(2340,dtype=torch.float64),BadNoise())
        self.assertTrue(bool(torch.isfinite(p).all()))
        spec=c.Spec((0,))
        with patch.object(mech,'center',side_effect=MemoryError('synthetic OOM')):
            with self.assertRaises(mech.ExecutionGap):mech.Engine.testing(spec,noise_seed=0).run(a.Graph((),(),True))

    def test_default_privacy_engines_share_composition_ledger(self):
        c,a,agg,acc,n,mech,r=self.api();spec=c.Spec(())
        first,second=mech.Engine.privacy(spec),mech.Engine.privacy(spec)
        self.assertIs(first._ledger,second._ledger)

    def test_graph_addition_is_inverse_of_deletion_and_ids_are_typed(self):
        c,a,*_=self.api();spec,graph=self.fixture()
        smaller=graph.delete(47)
        added=a.Graph(smaller.nodes+(47,),smaller.edges+tuple(e for e in graph.edges if 47 in e),True)
        self.assertEqual(a.Adapter(spec,graph).fingerprints(),a.Adapter(spec,added).fingerprints())
        with self.assertRaises(ValueError):a.Graph((True,),(),True)

    def test_production_rejects_test_rng_injection_and_second_model_before_graph(self):
        c,a,agg,acc,n,mech,r=self.api();spec=c.Spec(())
        instance=mech.Engine.privacy(spec,ledger=acc.Ledger());instance._source=n.TestGaussian(0)
        with self.assertRaises(TypeError):instance.run(a.Graph((),(),True))
        ledger=acc.Ledger();ledger.charge()
        instance=mech.Engine.privacy(spec,ledger=ledger)
        with patch.object(mech,'Adapter',side_effect=AssertionError('graph accessed')):
            with self.assertRaises(ValueError):instance.run(a.Graph((),(),True))

    def test_noise_source_exact_scale_and_all_coordinates_using_code_probe(self):
        c,a,agg,acc,n,mech,r=self.api()
        source=n.SecretGaussian()
        with patch.object(source._SecretGaussian__rng,'normalvariate',return_value=17.) as draw:
            packet=source.draw()
        self.assertEqual(draw.call_count,2340)
        self.assertTrue(all(call.args==(0.,260.) for call in draw.call_args_list))
        self.assertTrue(bool((packet==17).all()))
        self.assertEqual(source.calls,1)

    def test_test_noise_stream_is_independent_of_initialization_and_sampling(self):
        c,a,agg,acc,n,mech,r=self.api();spec,graph=self.fixture()
        source,control=n.TestGaussian(11),n.TestGaussian(11)
        self.assertTrue(torch.equal(source.draw(),control.draw()))
        mech.new_model();a.Adapter(spec,graph).batch()
        self.assertTrue(torch.equal(source.draw(),control.draw()))

    def test_nonfinite_real_model_gradient_is_zeroed_without_private_branch_release(self):
        c,a,agg,acc,n,mech,r=self.api();spec,graph=self.fixture()
        batch=a.Adapter(spec,graph).batch()
        self.assertTrue(any(item is not None for slot in batch for item in slot))
        with patch.object(mech,'read_gradient',return_value=torch.full((2340,),float('nan'))) as read:
            s,b=mech.center(mech.new_model(),batch)
        self.assertGreater(read.call_count,0)
        self.assertEqual(float(s.norm()),0.);self.assertEqual(float(b.norm()),0.)

    def test_ledger_counter_cannot_be_reset_through_public_property(self):
        _,_,_,acc,*_=self.api()
        ledger=acc.Ledger();ledger.charge()
        with self.assertRaises(AttributeError):ledger.models=0

    def test_no_real_dataset_loader_in_engine_or_cli(self):
        self.api()
        import ast
        from pathlib import Path
        package=Path(__file__).resolve().parents[1]/'experiments/node_dp_slots'
        for path in package.glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                    self.assertNotIn(name,('load_prepared_train','load_prepared_graph','load_saved_case','train_case'))

    def test_undirected_model_path_and_neighbor_contributions(self):
        c,a,agg,acc,n,mech,r=self.api();spec,graph=self.fixture(False)
        model=mech.new_model();left=a.Adapter(spec,graph);right=a.Adapter(spec,graph.delete(0))
        s,b=mech.center(model,left.batch());t,d=mech.center(model,right.batch())
        self.assertLessEqual(float((s-t).norm()),12.)
        for j in set(range(4))-set(c.memberships(0,0)):self.assertTrue(torch.equal(b[j],d[j]))
        result=mech.Engine.testing(spec,noise_seed=17).run(graph)
        self.assertEqual(result.testing_diagnostics['updates'],100)
        self.assertTrue(result.testing_diagnostics['finite_parameters'])

    def test_composition_charge_is_atomic_for_concurrent_callers(self):
        _,_,_,acc,*_=self.api()
        from concurrent.futures import ThreadPoolExecutor
        ledger=acc.Ledger()
        def attempt(_):
            try:ledger.charge();return True
            except ValueError:return False
        with ThreadPoolExecutor(max_workers=8) as pool:successes=list(pool.map(attempt,range(16)))
        self.assertEqual(sum(successes),1);self.assertEqual(ledger.models,1)


if __name__=='__main__':unittest.main()

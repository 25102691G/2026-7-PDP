"""Artificial data only: no prepared loader, private IDs or real-data training."""
import importlib.util
import itertools
import json
import math
import random
import unittest
from dataclasses import replace
from unittest.mock import patch


class NodePDPTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.node_pdp'))
        from experiments.node_pdp import registry, accounting, sampling, mechanism, release
        return registry, accounting, sampling, mechanism, release

    def fixture(self, directed=True):
        r, *_ = self.api()
        from experiments.node_dp_slots.config import Spec
        from experiments.node_dp_slots.adapter import Graph
        ids = (10, 30, 70, 90, 100)
        reg = r.Registry(ids, tuple(zip(ids, (1, 2, 4, 1, 2))), source='public-artificial-v1')
        return reg, Spec(ids, directed), Graph((90, 10, 70), ((10, 70),), directed)

    def test_registry_complete_immutable_and_stable(self):
        r, *_ = self.api(); reg, spec, graph = self.fixture()
        self.assertEqual(reg.universe, spec.universe)
        original = {v: reg.entry(v) for v in reg.universe}
        for v in graph.nodes:
            smaller = graph.delete(v)
            self.assertEqual({u: reg.entry(u) for u in smaller.nodes},
                             {u: original[u] for u in smaller.nodes})
        self.assertEqual(reg.entry(10).epsilon_target, 1)
        self.assertAlmostEqual(reg.entry(10).retention_probability, .2689414213699951)
        with self.assertRaises(Exception): reg.universe = ()
        with self.assertRaises(KeyError): reg.entry(999)
        for budgets in (((10, 1),), ((10, 1), (10, 2)), ((10, 3), (30, 2))):
            with self.assertRaises(ValueError): r.Registry((10, 30), budgets, source='public')
        with self.assertRaises(ValueError): r.Registry((True,), ((True, 1),), source='public')
        with self.assertRaises(ValueError): r.Registry((1,), ((1, 1),), source='')
        with self.assertRaises(ValueError): replace(reg, version='unknown')

    def test_public_quotas_only_depend_on_public_universe(self):
        r, *_ = self.api()
        a = r.Registry.from_public_quotas(tuple(range(10)), public_seed=4, source='public')
        b = r.Registry.from_public_quotas(tuple(reversed(range(10))), public_seed=4, source='public')
        self.assertEqual(a, b)
        self.assertEqual([sum(a.entry(v).epsilon_target == e for v in a.universe) for e in (1, 2, 4)], [2, 4, 4])
        self.assertEqual(a.fingerprint, b.fingerprint)
        self.assertNotEqual(a.fingerprint, replace(a, source='different-public-source').fingerprint)
        self.assertEqual(len(a.public()['entries']), 10)

    def test_accounting_reads_actual_frozen_base_and_both_directions(self):
        _, a, *_ = self.api(); reg, spec, _ = self.fixture()
        base = a.base_account(spec)
        self.assertAlmostEqual(base['epsilon'], 1.982092738991924)
        table = a.node_accounts(reg, spec)
        self.assertEqual(len(table), len(spec.universe))
        for row in table:
            self.assertTrue(row['target_satisfied'])
            self.assertLessEqual(row['epsilon_reverse'], row['epsilon_prime'] + 1e-14)
            self.assertLessEqual(row['delta_reverse'], row['delta_prime'] + 1e-20)
        low = table[0]
        # Actual base epsilon is not the rounded 1.982092739 reference.
        self.assertAlmostEqual(low['epsilon_prime'], .986940353796760885, places=14)
        self.assertAlmostEqual(low['delta_prime'], 2.6894142137e-6, places=16)
        from experiments.node_dp_slots import accounting as base_acc
        with patch.object(base_acc, 'epsilon', return_value=(1.5, 9.)) as probe:
            self.assertEqual(a.base_account(spec)['epsilon'], 1.5)
        probe.assert_called_once_with(std=260., models=1)

    def test_amplification_boundaries_and_invalid_numbers(self):
        _, a, *_ = self.api()
        self.assertEqual(a.amplify(2., 1e-5, 0.)['epsilon_prime'], 0.)
        self.assertEqual(a.amplify(2., 1e-5, 0.)['delta_prime'], 0.)
        self.assertAlmostEqual(a.amplify(2., 1e-5, 1.)['epsilon_prime'], 2.)
        self.assertAlmostEqual(a.amplify(2., 1e-5, 1.)['delta_prime'], 1e-5)
        for p in (-1., 1.1, float('nan')):
            with self.assertRaises(ValueError): a.amplify(2., 1e-5, p)

    def test_small_epsilon_accounting_avoids_cancellation(self):
        _, a, *_ = self.api()
        self.assertAlmostEqual(a.amplify(1e-20,1e-5,.5)['epsilon_prime']/1e-20,.5,places=14)

    def test_sampling_survivor_probabilities_are_stable_by_identity(self):
        _, _, s, *_=self.api();reg,spec,graph=self.fixture()
        class Probe:
            def __init__(self):self.calls=[]
            def draw(self,p):self.calls.append(p);return True
        first=Probe();s.sample_once(graph,reg,first)
        expected=dict(zip(sorted(graph.nodes),first.calls))
        for v in graph.nodes:
            second=Probe();smaller=graph.delete(v);s.sample_once(smaller,reg,second)
            self.assertEqual(dict(zip(sorted(smaller.nodes),second.calls)),
                             {u:p for u,p in expected.items() if u!=v})

    def test_dense_artificial_composite_has_real_gradients_and_fixed_batches(self):
        r,_,s,m,_=self.api()
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.config import Spec
        from experiments.node_dp_slots.adapter import Graph
        ids=tuple(range(160));reg=r.Registry(ids,tuple((v,(1,2,4,2)[v%4]) for v in ids),source='public-artificial-dense')
        graph=Graph(tuple(range(128)),tuple((v,(v+k)%128) for v in range(128) for k in range(1,17)),True)
        observed=[];center=base.center
        def checked(model,batch):
            self.assertEqual([len(slot) for slot in batch],[4]*4)
            signal,slots=center(model,batch)
            self.assertEqual(tuple(signal.shape),(2340,))
            self.assertTrue(all(float(vector.norm())<=3. for vector in slots))
            observed.append(float(signal.norm()))
            return signal,slots
        with patch.object(base,'center',side_effect=checked):
            result=m.Engine.testing(reg,Spec(ids),sampling_seed=5,noise_seed=5).run(graph)
        self.assertEqual(len(observed),100)
        self.assertGreater(max(observed),0.)
        self.assertTrue(result.testing_diagnostics['finite_parameters'])

    def test_configuration_disambiguates_base_and_personalized_targets(self):
        _,_,_,m,release=self.api();reg,spec,graph=self.fixture()
        with self.privacy_context():
            result=m.Engine.privacy(reg,spec).run(graph)
            cfg=release.export(result,permit_idealized=True)['configuration']
            self.assertNotIn('epsilon_target',cfg)
            self.assertNotIn('sampling_seed',cfg)
            self.assertNotIn('batch_seed',cfg)
            self.assertEqual(cfg['base_epsilon_target'],2.)
            self.assertEqual(cfg['base_sampling_seed'],0)
            self.assertEqual(cfg['base_account']['s_sum'],260.)

    def test_finite_discrete_mixtures_both_directions(self):
        _, a, *_ = self.api()
        # Enumerate all events of artificial 3-point kernels satisfying base DP.
        for q, r, eps, delta in (((.5,.3,.2),(.3,.4,.3),math.log(2),0.),
                                 ((.6,.3,.1),(.2,.3,.5),math.log(2),.3)):
            for bits in itertools.product((0,1), repeat=3):
                Q=sum(x*b for x,b in zip(q,bits)); R=sum(x*b for x,b in zip(r,bits))
                self.assertLessEqual(R, math.exp(eps)*Q+delta+1e-14)
                self.assertLessEqual(Q, math.exp(eps)*R+delta+1e-14)
                for p in (0., .2689414213699951, .6, 1.):
                    row=a.amplify(eps,delta,p); P=(1-p)*Q+p*R
                    self.assertLessEqual(P,math.exp(row['epsilon_prime'])*Q+row['delta_prime']+1e-14)
                    self.assertLessEqual(Q,math.exp(row['epsilon_prime'])*P+row['delta_prime']+1e-14)

    def test_sampling_induced_graph_direction_isolates_and_local_map(self):
        _, _, s, *_ = self.api(); reg, spec, graph = self.fixture()
        for directed in (True, False):
            graph=replace(graph,directed=directed)
            kept=s.induce(graph,(10,70,90))
            self.assertEqual(kept.graph.nodes,(10,70,90))
            self.assertEqual(kept.graph.edges,((10,70),))
            self.assertEqual(kept.graph.directed,directed)
            self.assertEqual(kept.local_to_stable,(10,70,90))
            self.assertEqual(kept.stable_to_local,{10:0,70:1,90:2})
        self.assertEqual(s.induce(graph,()).graph.nodes,())

    def test_sampling_coupling_retains_node_adjacency(self):
        _, _, s, *_ = self.api(); reg, spec, graph = self.fixture()
        for target in graph.nodes:
            for bits in itertools.product((False,True),repeat=len(graph.nodes)):
                selected={v for v,b in zip(graph.nodes,bits) if b}
                left=s.induce(graph,selected).graph
                right=s.induce(graph.delete(target),selected-{target}).graph
                self.assertEqual(left.delete(target),right)

    def test_independent_bernoulli_no_topup_and_boundary_probabilities(self):
        _, _, s, *_ = self.api()
        source=s.TestBernoulli(41)
        self.assertFalse(source.draw(0.)); self.assertTrue(source.draw(1.))
        for p in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError): source.draw(p)
        # Finite statistical check, not an independence proof.
        pairs=[(source.draw(.27),source.draw(.63)) for _ in range(16000)]
        x=sum(a for a,b in pairs)/len(pairs); y=sum(b for a,b in pairs)/len(pairs)
        joint=sum(a and b for a,b in pairs)/len(pairs)
        self.assertLess(abs(x-.27),.015); self.assertLess(abs(y-.63),.015)
        self.assertLess(abs(joint-x*y),.01)

    def test_secret_sampler_uses_separate_entropy_and_no_seed(self):
        _, _, s, *_ = self.api()
        with self.assertRaises(TypeError): s.SecretBernoulli(seed=1)
        source=s.SecretBernoulli(); control=s.TestBernoulli(12)
        state=random.getstate(); source.draw(.3)
        self.assertEqual(state,random.getstate())
        self.assertTrue(hasattr(source,'draw'))
        # Counter/noise state from either implementation never enters public config.
        self.assertEqual(control.draw(.3),s.TestBernoulli(12).draw(.3))

    def test_secret_coin_integer_scale_and_one_draw_per_existing_identity(self):
        _,_,s,*_=self.api();reg,spec,graph=self.fixture()
        source=s.SecretBernoulli()
        with patch.object(source._SecretBernoulli__rng,'getrandbits',return_value=0) as entropy:
            selected=s.sample_once(graph,reg,source)
        self.assertEqual(entropy.call_count,len(graph.nodes))
        self.assertTrue(all(call.args==(64,) for call in entropy.call_args_list))
        self.assertEqual(selected.graph.nodes,tuple(sorted(graph.nodes)))
        self.assertLessEqual(s._threshold(.2689414213699951)/(1<<64),.2689414213699951)

    def test_private_invalid_domain_is_generic_and_public_configuration_cannot_recalibrate(self):
        _,_,_,m,_=self.api();reg,spec,graph=self.fixture()
        from experiments.node_dp_slots.config import Spec
        for bad in (replace(graph,nodes=(999,),edges=()),replace(graph,edges=((10,10),)),
                    replace(graph,edges=((10,70),(10,70))),replace(graph,directed=False)):
            with self.assertRaises(m.ExecutionGap):
                m.Engine.testing(reg,spec,sampling_seed=0,noise_seed=0).run(bad)
        with patch.object(Spec,'s_sum',261.):
            with self.assertRaises(ValueError):m.Engine.testing(reg,spec,sampling_seed=0,noise_seed=0)

    def test_public_mismatch_rejected_before_graph(self):
        _, _, _, m, _ = self.api(); reg, spec, graph = self.fixture()
        with self.assertRaises(ValueError): m.Engine.testing(reg,replace(spec,universe=(10,)),sampling_seed=0,noise_seed=0)
        with self.assertRaises(TypeError): m.Engine.privacy(reg,spec,sampling_seed=0)

    def test_testing_full_100_steps_once_and_no_release(self):
        _, _, s, m, release = self.api(); reg, spec, graph = self.fixture()
        engine=m.Engine.testing(reg,spec,sampling_seed=0,noise_seed=7)
        with patch.object(s,'sample_once',wraps=s.sample_once) as sample:
            result=engine.run(graph)
        self.assertEqual(sample.call_count,1)
        self.assertEqual(result.testing_diagnostics['updates'],100)
        self.assertEqual(result.testing_diagnostics['noise_calls'],100)
        self.assertEqual(len(result._base_run._parameters),2340)
        self.assertEqual(result._base_run._spec,spec)
        with self.assertRaises(PermissionError): release.export(result,permit_idealized=True)
        with self.assertRaises(PermissionError): engine.run(graph)

    def test_all_dropped_and_empty_still_run_100_steps(self):
        _, _, s, m, _ = self.api(); reg, spec, graph = self.fixture()
        # Both actual nodes have p=.2689; fixed test coins reject both legitimately.
        graph=replace(graph,nodes=(10,90),edges=((10,90),))
        self.assertEqual(s.sample_once(graph,reg,s.TestBernoulli(2)).graph.nodes,())
        result=m.Engine.testing(reg,spec,sampling_seed=2,noise_seed=2).run(graph)
        self.assertEqual(result.testing_diagnostics['updates'],100)
        result=m.Engine.testing(reg,spec,sampling_seed=2,noise_seed=2).run(replace(graph,nodes=(),edges=()))
        self.assertEqual(result.testing_diagnostics['updates'],100)

    def privacy_context(self):
        # Reset guards only under explicit test patch; production has no reset API.
        from experiments.node_dp_slots import mechanism as base, accounting as acc
        from experiments.node_pdp import mechanism as m
        from contextlib import ExitStack
        stack=ExitStack(); stack.enter_context(patch.object(base,'_PROCESS_LEDGER',acc.Ledger()))
        stack.enter_context(patch.object(m,'_PROCESS_GUARD',m._SingleUseGuard()))
        return stack

    def test_privacy_export_whitelist_strict_gate_and_single_release(self):
        _, _, _, m, release=self.api();reg,spec,graph=self.fixture()
        with self.privacy_context():
            engine=m.Engine.privacy(reg,spec); result=engine.run(graph)
            self.assertIsNone(result.testing_diagnostics)
            with self.assertRaises(PermissionError):release.export(result)
            payload=release.export(result,permit_idealized=True)
            self.assertEqual(set(payload),{'parameters','configuration'})
            self.assertEqual(len(payload['parameters']),2340)
            cfg=payload['configuration'];self.assertFalse(cfg['machine_dp_certified'])
            self.assertEqual(cfg['registry'],reg.public())
            self.assertEqual(cfg['s_sum'],260.)
            for forbidden in ('retained_nodes','retained_count','decisions','sampling_seed','batch_seed',
                              'private_mapping','cache','loss','noise_seed'):
                self.assertNotIn(forbidden,cfg.get('outer',{}))
                self.assertNotIn(forbidden,payload)
            with self.assertRaises(PermissionError):release.export(result,permit_idealized=True)
            with self.assertRaises(PermissionError):engine.run(graph)
            with self.assertRaises(PermissionError):m.Engine.privacy(reg,spec).run(None)

    def test_budget_reserved_before_sampling_and_default_base_ledger_agrees(self):
        _, _, s, m, _=self.api();reg,spec,graph=self.fixture()
        from experiments.node_dp_slots import mechanism as base
        with self.privacy_context():
            engine=m.Engine.privacy(reg,spec)
            original=s.sample_once
            def inspect(*args):
                self.assertEqual(base._PROCESS_LEDGER.models,1)
                return original(*args)
            with patch.object(s,'sample_once',side_effect=inspect):engine.run(graph)
            self.assertEqual(base._PROCESS_LEDGER.models,1)
            with self.assertRaises(ValueError):base.Engine.privacy(spec).run(None)

    def test_failure_burns_reservation_generic_error_no_private_details(self):
        _, _, s, m, release=self.api();reg,spec,graph=self.fixture()
        from experiments.node_dp_slots import mechanism as base
        with self.privacy_context():
            engine=m.Engine.privacy(reg,spec)
            with patch.object(s,'sample_once',side_effect=RuntimeError('secret nodes=999')):
                with self.assertRaises(m.ExecutionGap) as caught:engine.run(graph)
            self.assertNotIn('999',str(caught.exception))
            self.assertEqual(base._PROCESS_LEDGER.models,1)
            with self.assertRaises(PermissionError):m.Engine.privacy(reg,spec).run(None)

    def test_privacy_rejects_injected_testing_sources(self):
        _, _, s, m, release=self.api();reg,spec,_=self.fixture()
        from experiments.node_dp_slots.noise import TestGaussian
        for which in ('outer','inner'):
            with self.privacy_context():
                engine=m.Engine.privacy(reg,spec)
                if which=='outer':engine._sampler=s.TestBernoulli(0)
                else:engine._base._source=TestGaussian(0)
                with self.assertRaises(PermissionError):engine.run(None)

    def test_concurrent_default_guard_allows_one_reservation(self):
        _, _, s, m, _=self.api();reg,spec,graph=self.fixture()
        from concurrent.futures import ThreadPoolExecutor
        with self.privacy_context():
            def attempt(_):
                try:m.Engine.privacy(reg,spec).run(graph);return True
                except PermissionError:return False
            with ThreadPoolExecutor(max_workers=4) as pool: successes=list(pool.map(attempt,range(4)))
            self.assertEqual(sum(successes),1)

    def test_no_prepared_loader_or_logging_or_old_accountant(self):
        self.api()
        import ast
        from pathlib import Path
        for path in (Path(__file__).resolve().parents[1]/'experiments/node_pdp').glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
                    self.assertNotIn(name,('load_prepared_train','load_prepared_graph','train_case','print','basicConfig'))


if __name__=='__main__':unittest.main()

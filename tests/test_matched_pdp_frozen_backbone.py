"""Artificial-only contract tests. No prepared files or historical checkpoints."""
import importlib.util
import math
import random
import unittest
from decimal import Decimal, localcontext
from unittest.mock import patch

import torch

PACKAGE = 'experiments.matched_pdp_frozen_backbone'


class FrozenBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec(PACKAGE), 'independent mechanism missing')
        from experiments.matched_pdp_frozen_backbone import config, layout, accounting, noise, kernel, research
        return config, layout, accounting, noise, kernel, research

    def fixture(self, directed=True, heterogeneous=True):
        config, *_ = self.api()
        from experiments.node_pdp.registry import Registry
        from experiments.node_dp_slots.adapter import Graph
        ids = tuple(range(12))
        reg = Registry(ids, tuple((v, (1, 2, 4)[v % 3] if heterogeneous else 2) for v in ids), source='public-artificial')
        edges = [(i, (i+1) % 10) for i in range(10)] + [(0, 3), (4, 2)]
        if not directed:
            edges = sorted({tuple(sorted(e)) for e in edges})
        return reg, config.Spec(ids, directed), Graph(ids, tuple(edges), directed)

    def rich_fixture(self, directed):
        config, *_ = self.api()
        from experiments.node_pdp.registry import Registry
        from experiments.node_dp_slots.adapter import Graph
        ids=tuple(range(128))
        reg=Registry(ids,tuple((v,2) for v in ids),source='public-artificial-dense')
        edges=[(v,(v+k)%96) for v in range(96) for k in range(1,13)]
        if not directed:edges=sorted({tuple(sorted(e)) for e in edges})
        return reg,config.Spec(ids,directed),Graph(tuple(range(96)),tuple(edges),directed)

    def test_seeds_layout_frozen_and_original_initialization(self):
        c, l, *_ = self.api()
        from experiments.node_dp_slots.mechanism import new_model
        for seed in range(5):
            original = new_model(seed)
            model = l.new_model(seed)
            layout = l.Layout.from_model(model)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 2340)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 33)
            self.assertEqual(layout.total, 33)
            self.assertEqual(layout.names, ('grat.grat3.fc.weight', 'grat.grat3.fc.bias'))
            for (name, a), (name2, b) in zip(model.named_parameters(), original.named_parameters()):
                self.assertEqual(name, name2)
                self.assertTrue(torch.equal(a, b))
                self.assertEqual(a.requires_grad, name in layout.names)
            self.assertTrue(bool((model.grat.grat3.fc.weight < 0).any()))
            self.assertEqual(c.Spec((0,)).public()['dimension'], 33)
        from experiments.node_dp_slots.config import Spec as Old
        self.assertEqual((Old((0,)).dimension, Old((0,)).s_sum), (2340, 260.))
        with self.assertRaises(TypeError): c.validate(Old((0,)))

    def test_head_gradients_and_forward_match_full_original(self):
        _, l, _, _, k, _ = self.api()
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_dp_slots.adapter import Item
        for directed in (True, False):
            for edges in ((), ((0, 1), (1, 2))):
                for seed in range(5):
                    a, b = l.new_model(seed), base.new_model(seed)
                    _, Loss, get_dgl, get_adj, ig = base.runtime()
                    item = Item(tuple(range(4)), edges, directed)
                    graph = ig.Graph(n=4, edges=edges, directed=directed)
                    message = get_dgl(graph, 'ones', 32, directed, False)
                    self.assertEqual(message.num_edges(), len(edges)*(1 if directed else 2))
                    self.assertTrue(torch.equal(a(message, message.ndata['feat']), b(message, message.ndata['feat'])))
                    if not directed:
                        graph = ig.Graph(n=4, edges=list(edges)+[(v,u) for u,v in edges], directed=True)
                    out = b(message, message.ndata['feat']).squeeze(1)
                    Loss(.5)(out, get_adj(graph), 50).backward()
                    expected = torch.cat([p.grad.reshape(-1).double() for n,p in b.named_parameters() if n in l.HEAD_NAMES])
                    self.assertTrue(torch.equal(k.position_gradient(a, item), expected))
                    self.assertTrue(all(p.grad is None for n,p in a.named_parameters() if n not in l.HEAD_NAMES))

    def test_double_clip_and_adversarial_twelve_bound(self):
        _, _, _, _, k, _ = self.api()
        from experiments.node_dp_slots.aggregation import clip
        positions = torch.stack([clip(torch.full((33,), 1e6, dtype=torch.float64))]*16).reshape(4,4,33)
        signal, slots = k.double_clip(positions)
        self.assertTrue(all(float(v.norm()) <= 3 for v in slots))
        other = slots.clone(); other[:2] *= -1
        self.assertLessEqual(float((signal-other.sum(0)).norm()), 12.)
        self.assertAlmostEqual(float((signal-other.sum(0)).norm()), 12., delta=2e-6)
        with self.assertRaises(ValueError): k.double_clip(torch.zeros(4,4,2340))
        bad = torch.ones(33); bad[0] = float('nan')
        self.assertTrue(torch.equal(clip(bad), torch.zeros(33, dtype=torch.float64)))

    def test_public_model_seed_is_immutable_provenance(self):
        *_,r=self.api()
        reg,spec,_=self.fixture()
        session=r.Session(reg,spec,model_seed=2)
        with self.assertRaises(AttributeError):session.model_seed=4
        self.assertEqual(session.model_seed,2)

    def test_secret_gaussian_exact_dimension_scale_calls(self):
        _, _, _, noise, *_ = self.api()
        calls = []
        with patch.object(random.SystemRandom, 'normalvariate', side_effect=lambda m,s: (calls.append((m,s)) or 1.)):
            source = noise.SecretGaussian(); v = source.draw()
        self.assertEqual(calls, [(0.,260.)]*33)
        self.assertEqual(v.shape, (33,)); self.assertEqual(v.dtype, torch.float64)
        self.assertEqual(source.calls, 1)

    def test_update_formula_and_only_weight_projection(self):
        _, l, _, _, k, _ = self.api()
        model = l.new_model(0); layout = l.Layout.from_model(model)
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        signal = torch.arange(33, dtype=torch.float64)/3
        z = torch.linspace(-260, 260, 33, dtype=torch.float64)
        class Fixed:
            calls = 0
            def draw(self): self.calls += 1; return z.clone()
        source = Fixed()
        old = torch.cat([before[n].reshape(-1).double() for n in l.HEAD_NAMES])
        expected = (old-.005*(signal+z)/16).float(); expected[:32].clamp_min_(0)
        k.step(model, signal, source, layout)
        self.assertEqual(source.calls, 1)
        self.assertTrue(torch.equal(l.head_vector(model).float(), expected))
        for n,p in model.named_parameters():
            if n not in l.HEAD_NAMES: self.assertTrue(torch.equal(p, before[n]))
        # Bias is allowed to become negative and all zero-gradient coordinates receive noise.
        self.assertLess(float(model.grat.grat3.fc.bias), 0.)
        with self.assertRaises(ValueError): k.step(model, torch.zeros(2340), source, layout)

    def test_independent_full_account_and_decimal_conversion(self):
        _, _, a, *_ = self.api(); reg, spec, _ = self.fixture()
        with localcontext() as ctx:
            ctx.prec = 55
            candidates=[]
            for i in range(11,1000):
                alpha=Decimal(i)/10
                rho=100*alpha*144/(2*Decimal(260)**2)
                eps=rho-(Decimal('1e-5').ln()+alpha.ln())/(alpha-1)+((alpha-1)/alpha).ln()
                candidates.append((eps,alpha))
            eps,alpha=min(candidates)
            p=(Decimal(1).exp()-1)/(Decimal(2).exp()-1)
            amplified=(1+p*(eps.exp()-1)).ln()
        with patch('experiments.node_pdp.accounting.base_account', side_effect=AssertionError('33D accountant must be independent')):
            result=a.base_account(spec); rows=a.node_accounts(reg,spec)
        self.assertAlmostEqual(result['epsilon'],float(eps),places=13)
        self.assertEqual(result['rdp_order'],float(alpha)); self.assertEqual(result['dimension'],33)
        self.assertEqual(result['T'],100)
        self.assertAlmostEqual(rows[0]['epsilon_prime'],float(amplified),places=13)
        self.assertAlmostEqual(rows[0]['delta_prime'],float(p)*1e-5,places=18)
        self.assertTrue(all(row['target_satisfied'] for row in rows))
        for p in (0.,.2689414213699951,1.):
            r=a.amplify(result['epsilon'],1e-5,p)
            self.assertLessEqual(r['epsilon_reverse'],r['epsilon_prime']+1e-14)
            self.assertLessEqual(r['delta_reverse'],r['delta_prime']+1e-18)

    def test_common_history_slot_locality_directed_undirected_and_isolates(self):
        _, l, _, _, k, _ = self.api()
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_dp_slots.config import memberships
        for directed in (True,False):
            reg,spec,graph=self.rich_fixture(directed)
            for v in (0,47,95):
                a,b=Adapter(spec,graph),Adapter(spec,graph.delete(v))
                model=l.new_model(1)
                for _ in range(2):
                    x,bx=k.center(model,a.batch()); y,by=k.center(model,b.batch())
                    for j in set(range(4))-set(memberships(v)):
                        self.assertTrue(torch.equal(bx[j],by[j]))
                    self.assertLessEqual(float((x-y).norm()),12.)
                    # Prescribed common noisy history; no coupling of actual private noises assumed.
                    with torch.no_grad(): model.grat.grat3.fc.bias.add_(.1)

    def test_full_hundred_step_reference_and_backbone_invariant_each_step(self):
        _, l, _, _, k, r = self.api()
        from experiments.matched_pdp_frozen_backbone.testing import TestGaussian
        from experiments.node_dp_slots.adapter import Adapter
        from experiments.node_pdp import sampling
        for directed in (True,False):
            reg,spec,graph=self.rich_fixture(directed)
            for seed in range(5):
                session=r.Session.synthetic(reg,spec,model_seed=seed,sampling_seed=12,noise_seed=21)
                original={n:p.detach().clone() for n,p in session.model.named_parameters()}
                ref=l.new_model(seed); layout=l.Layout.from_model(ref)
                induced=sampling.sample_once(graph,reg,sampling.TestBernoulli(12))
                adapter=Adapter(spec,induced.graph); source=TestGaussian(21)
                self.assertTrue(any(c.subgraphs for c in adapter.caches), 'must exercise actual subgraph gradients')
                saw_signal=False
                for _ in range(100):
                    signal,slots=k.center(ref,adapter.batch()); k.step(ref,signal,source,layout)
                    saw_signal |= bool(torch.count_nonzero(signal))
                self.assertTrue(saw_signal, 'test must exercise nonzero clipped head gradients')
                real_step=k.step
                def checked(model,*args):
                    stat=real_step(model,*args)
                    for n,p in model.named_parameters():
                        if n not in l.HEAD_NAMES:self.assertTrue(torch.equal(p,original[n]))
                    return stat
                with patch.object(k,'step',side_effect=checked) as steps:
                    result=session.run(graph)
                self.assertEqual(steps.call_count,100)
                self.assertEqual(result.parameters,tuple(float(v) for v in torch.cat([p.detach().reshape(-1) for p in ref.parameters()])))
                self.assertEqual(result.diagnostics['noise_coordinates'],3300)
                self.assertEqual(result.diagnostics['updates'],100)
                self.assertEqual(result.diagnostics['outer_sampling_calls'],1)
                self.assertEqual(result.diagnostics['projection_calls'],100)
                self.assertTrue(result.diagnostics['frozen_backbone_unchanged'])
                self.assertFalse(result.configuration['publication_allowed'])
                with self.assertRaises(PermissionError):session.run(graph)

    def test_empty_and_isolate_only_graphs_never_skip_noise(self):
        _, _, _, _, _, r=self.api()
        from experiments.node_dp_slots.adapter import Graph
        reg,spec,_=self.fixture()
        for nodes in ((),(0,),(0,1,2)):
            result=r.Session.synthetic(reg,spec,sampling_seed=0,noise_seed=0).run(Graph(nodes,(),True))
            self.assertEqual(result.diagnostics['noise_calls'],100)
            self.assertEqual(result.diagnostics['noise_coordinates'],3300)
            self.assertTrue(result.diagnostics['all_projected_weights_nonnegative'])

    def test_scientific_path_rejects_test_sources_and_modified_initial_state(self):
        _, _, _, _, _, r=self.api()
        from experiments.matched_pdp_frozen_backbone.testing import TestGaussian
        from experiments.node_pdp.sampling import TestBernoulli
        reg,spec,g=self.fixture()
        for kind in ('noise','sampler','head','backbone'):
            session=r.Session(reg,spec)
            if kind=='noise': session._noise=TestGaussian(0)
            elif kind=='sampler': session._sampler=TestBernoulli(0)
            else:
                parameter=session.model.grat.grat3.fc.bias if kind=='head' else session.model.grat.grat1.fc.weight
                with torch.no_grad():parameter.add_(1)
            with self.assertRaises((PermissionError,ValueError)):session.run(g)

    def test_research_uses_independent_os_sources_no_ledger_or_secret_output(self):
        _, _, _, noise, _, r=self.api()
        from experiments.node_dp_slots import mechanism as base
        from experiments.node_pdp import mechanism as v1, sampling
        from experiments.node_dp_slots.adapter import Graph
        import dataclasses,json
        reg,spec,g=self.fixture(); before=(base._PROCESS_LEDGER.models,v1._PROCESS_GUARD._used)
        session=r.Session(reg,spec,model_seed=3)
        self.assertIs(type(session._noise),noise.SecretGaussian)
        self.assertIs(type(session._sampler),sampling.SecretBernoulli)
        result=session.run(Graph((),(),True))
        self.assertEqual(before,(base._PROCESS_LEDGER.models,v1._PROCESS_GUARD._used))
        text=json.dumps(dataclasses.asdict(result))
        for token in ('noise_seed','sampling_seed','random_bytes','retained_mask','noise_vector','rng_state'):
            self.assertNotIn(token,text)
        for path in ('experiments.node_dp_slots.release','experiments.node_pdp.release','experiments.node_pdp_v2_projected.release','experiments.matched_pdp_frozen_backbone.release'):
            from importlib import import_module
            with self.assertRaises(PermissionError):import_module(path).export(result,permit_idealized=True)


if __name__=='__main__':unittest.main()

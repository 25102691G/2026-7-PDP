"""Read-only original-forward counterexample diagnosis on artificial graphs."""
import unittest
import torch


class EmptyEdgeDiagnostic(unittest.TestCase):
    def test_original_first_aggregate_and_frozen_q1(self):
        from experiments.node_dp_slots.mechanism import runtime,new_model
        from experiments.rowsum_pdp.model import Model
        from pathlib import Path
        import json
        torch.set_num_threads(1)
        records=[]
        for n,edges in [(4,()),(4,((0,1),)),(0,())]:
            graph=runtime()[4].Graph(n=n,edges=edges,directed=True)
            message=runtime()[2](graph,'ones',32,True,False)
            old=new_model(0);new=Model(0);capture={}
            hook=old.grat.grat1.fc.register_forward_pre_hook(lambda module,args:capture.update(aggregate=args[0].detach().clone()))
            with torch.no_grad():
                old_values=old.logits(message,message.ndata['feat']);new_values=new.logits(message,message.ndata['feat'])
            hook.remove()
            agg=capture['aggregate'];q=new.q1(message,message.ndata['feat'])
            expected=q[:,None].expand(n,32)
            record=dict(nodes=n,edges=len(edges),first_aggregate_shape=list(agg.shape),
                original_first_aggregate_equals_q1=bool(torch.equal(agg,expected)),
                original_aggregate_unique_values=sorted(set(float(x) for x in agg.reshape(-1))),
                q1_values=q.tolist(),original_logits=old_values.reshape(-1).tolist(),candidate_logits=new_values.reshape(-1).tolist(),
                max_absolute_logit_difference=float((old_values-new_values).abs().max()) if n else 0.)
            records.append(record)
        p=Path('docs/pdp_research/stage_c2_rowsum/empty_edge_counterexample.json')
        p.write_text(json.dumps(records,indent=2)+'\n');p.chmod(0o600)
        self.assertFalse(records[0]['original_first_aggregate_equals_q1'])
        self.assertEqual(records[0]['original_aggregate_unique_values'],[1.])
        self.assertEqual(records[0]['q1_values'],[0.,0.,0.,0.])
        self.assertTrue(records[1]['original_first_aggregate_equals_q1'])
        self.assertTrue(records[2]['original_first_aggregate_equals_q1'])


if __name__=='__main__':unittest.main()


class QuarantineDiagnostic(unittest.TestCase):
    def test_candidate_session_blocked_before_private_or_test_draw(self):
        from unittest.mock import patch
        from test_rowsum_pdp import RowSumTests
        from experiments.rowsum_pdp.research import Session
        registry,spec,graph=RowSumTests().fixture()
        session=Session.synthetic(registry,spec,model_seed=0,sampling_seed=0,noise_seed=0)
        with patch.object(session._noise,'draw',side_effect=AssertionError('must not draw')):
            with self.assertRaisesRegex(PermissionError,'C2 BLOCKED'):
                session.run(graph)
        self.assertEqual(session._noise.calls,0)
        with self.assertRaisesRegex(PermissionError,'C2 BLOCKED'):
            Session(registry,spec,model_seed=0).run(None)

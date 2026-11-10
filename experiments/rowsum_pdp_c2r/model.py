"""Minimal correction to the frozen1283 representation; no old source changes."""
import torch
from experiments.rowsum_pdp.model import Model as OriginalRepresentation,Layout


class Model(OriginalRepresentation):
    @staticmethod
    def q1(graph,features):
        # The inherited validator and nonempty-edge normalization read only this F.
        q=OriginalRepresentation.q1(graph,features)
        return torch.ones_like(q) if graph.num_edges()==0 else q

    def logits(self,graph,features):
        # Keep the original empty-tensor autograd path as well as its empty values.
        h=self.first_hidden(graph,features)
        return self.layer3(graph,torch.relu(self.layer2(graph,h)))

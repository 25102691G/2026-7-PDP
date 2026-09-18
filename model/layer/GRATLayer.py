import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.softmax import edge_softmax


GRAT_INITIALIZATIONS = ("released_default", "structural_preserving")


def _apply_initialization(fc, attn_fc, initialization):
    if initialization == "released_default":
        return
    if initialization != "structural_preserving":
        raise ValueError(f"unsupported GRAT initialization: {initialization}")
    # All released experiments use non-negative all-one inputs followed by
    # ReLU. Random signed row sums can invert or erase the only initial
    # structural signal. Preserve random magnitudes while making each row a
    # positive unit-sum aggregation. Start attention close to uniform but in
    # ReLU's differentiable positive region.
    with torch.no_grad():
        fc.weight.abs_()
        fc.weight.div_(fc.weight.sum(dim=1, keepdim=True).clamp_min(1e-12))
        if fc.bias is not None:
            fc.bias.zero_()
        nn.init.normal_(attn_fc.weight, mean=0.0, std=0.01)
        if attn_fc.bias is not None:
            attn_fc.bias.fill_(0.1)


class GRATLayer(nn.Module):
    """Paper GRAT layer from Appendix G, Equations (37)-(40)."""

    def __init__(self, in_feats, out_feats,
                 initialization="released_default"):
        # TODO: why not another linear layer before entering the net
        super(GRATLayer, self).__init__()
        # linear layer
        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        
        # attention layer
        self.attn_fc = nn.Linear(2 * out_feats, 1, bias=False)
        _apply_initialization(self.fc, self.attn_fc, initialization)

        # initialize parameters
        # self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        h2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(h2)
        return {'e': F.leaky_relu(a, negative_slope=0.2)}

    def message_func(self, edges):
        return {'h': edges.src['z'] * edges.data['alpha']}

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['z'] = self.fc(feature)
            # Appendix G, Equations (39)-(40).
            g.apply_edges(self.edge_attention)  # calculate e_{ij}

            # Calculate softmax on source code -> on the reversed graph
            rg = g.reverse(copy_ndata=False, copy_edata=True)
            g.edata['alpha'] = edge_softmax(rg, rg.edata['e'])
            
            # Convolution
            g.update_all(self.message_func, self.reduce_func)
            
            h = g.ndata['h']
            return h


class LegacyGRATLayer(nn.Module):
    """Released-code GRAT layer retained for old checkpoint compatibility."""

    def __init__(self, in_feats, out_feats,
                 initialization="released_default"):
        super().__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=True)
        self.attn_fc = nn.Linear(2 * in_feats, 1, bias=True)
        _apply_initialization(self.fc, self.attn_fc, initialization)

    def edge_attention(self, edges):
        h2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {'e': torch.relu(self.attn_fc(h2))}

    def message_func(self, edges):
        return {'h': edges.src['h'] * edges.data['alpha']}

    def reduce_func(self, nodes):
        return {'h': torch.sum(nodes.mailbox['h'], dim=1)}

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.apply_edges(self.edge_attention)
            rg = g.reverse(copy_ndata=False, copy_edata=True)
            g.edata['alpha'] = edge_softmax(rg, rg.edata['e'])
            g.update_all(self.message_func, self.reduce_func)
            return self.fc(g.ndata['h'])


class LegacyLeakyGRATLayer(LegacyGRATLayer):
    """Released GRAT parameterization with Appendix G attention activation.

    This diagnostic changes only the released layer's edge-score activation
    from ReLU to LeakyReLU (Appendix G, Equation 40).  Its parameter shapes,
    biased linear maps, source normalization and aggregation stay identical
    to ``LegacyGRATLayer`` so the effect can be isolated.
    """

    def edge_attention(self, edges):
        h2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {
            'e': F.leaky_relu(self.attn_fc(h2), negative_slope=0.2)
        }

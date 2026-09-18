from model.layer.GRATLayer import (
    GRATLayer,
    LegacyGRATLayer,
    LegacyLeakyGRATLayer,
)
from model.layer.GRATVarLayer import GRATVLayer
import torch.nn as nn
import torch


PAPER_GRAT = "paper_appendix_g"
LEGACY_GRAT = "legacy_released"
LEGACY_LEAKY_GRAT = "legacy_leaky_attention"
GRAT_VARIANTS = (PAPER_GRAT, LEGACY_GRAT, LEGACY_LEAKY_GRAT)


def _grat_layer(variant):
    if variant == PAPER_GRAT:
        return GRATLayer
    if variant == LEGACY_GRAT:
        return LegacyGRATLayer
    if variant == LEGACY_LEAKY_GRAT:
        return LegacyLeakyGRATLayer
    raise ValueError(f"unsupported GRAT variant: {variant}")


# grat models
# List of models: GRAT2, GRAT3, GRAT4

class GRAT2_(nn.Module):  # GRAT2 before sigmoid
    def __init__(self, in_feats, hidden_feats1, out_feats, *args,
                 variant=LEGACY_GRAT, initialization="released_default"):
        super(GRAT2_, self).__init__()
        layer = _grat_layer(variant)
        self.grat1 = layer(in_feats, hidden_feats1, initialization)
        self.grat2 = layer(hidden_feats1, out_feats, initialization)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))
        h = self.grat2(g, h)
        return h

class GRAT2(nn.Module):
    def __init__(self, in_feats, hidden_feats1, *args, variant=LEGACY_GRAT,
                 initialization="released_default"):
        super(GRAT2, self).__init__()
        self.grat = GRAT2_(
            in_feats, hidden_feats1, 1, *args, variant=variant,
            initialization=initialization,
        )

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRAT3_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, out_feats, *args,
                 variant=LEGACY_GRAT, initialization="released_default"):
        super(GRAT3_, self).__init__()
        layer = _grat_layer(variant)
        self.grat1 = layer(in_feats, hidden_feats1, initialization)
        self.grat2 = layer(hidden_feats1, hidden_feats2, initialization)
        self.grat3 = layer(hidden_feats2, out_feats, initialization)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))
        h = torch.relu(self.grat2(g, h))
        h = self.grat3(g, h)
        return h


class GRAT3(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, *args,
                 variant=LEGACY_GRAT, initialization="released_default"):
        super(GRAT3, self).__init__()
        self.grat = GRAT3_(
            in_feats, hidden_feats1, hidden_feats2, 1, *args, variant=variant,
            initialization=initialization,
        )  # out_feats=1

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRAT4_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3,
                 out_feats, *args, variant=LEGACY_GRAT,
                 initialization="released_default"):
        super(GRAT4_, self).__init__()
        layer = _grat_layer(variant)
        self.grat1 = layer(in_feats, hidden_feats1, initialization)
        self.grat2 = layer(hidden_feats1, hidden_feats2, initialization)
        self.grat3 = layer(hidden_feats2, hidden_feats3, initialization)
        self.grat4 = layer(hidden_feats3, out_feats, initialization)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))  # Guanhao uses tanh
        h = torch.relu(self.grat2(g, h))
        h = torch.relu(self.grat3(g, h))
        h = self.grat4(g, h)
        return h


class GRAT4(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3,
                 *args, variant=LEGACY_GRAT,
                 initialization="released_default"):
        super(GRAT4, self).__init__()
        self.grat = GRAT4_(
            in_feats, hidden_feats1, hidden_feats2, hidden_feats3, 1, *args,
            variant=variant, initialization=initialization,
        )

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRATV2_(nn.Module):  # GRAT2 before sigmoid
    def __init__(self, in_feats, hidden_feats1, out_feats, *args):
        super(GRATV2_, self).__init__()
        self.grat1 = GRATVLayer(in_feats, hidden_feats1)
        self.grat2 = GRATVLayer(hidden_feats1, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))
        h = self.grat2(g, h)
        return h

class GRATV2(nn.Module):
    def __init__(self, in_feats, hidden_feats1, *args):
        super(GRATV2, self).__init__()
        self.grat = GRATV2_(in_feats, hidden_feats1, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRATV3_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, out_feats, *args):
        super(GRATV3_, self).__init__()
        self.grat1 = GRATVLayer(in_feats, hidden_feats1)
        self.grat2 = GRATVLayer(hidden_feats1, hidden_feats2)
        self.grat3 = GRATVLayer(hidden_feats2, out_feats)

    def forward(self, g, feature):
        h = torch.tanh(self.grat1(g, feature))  # TODO: Guanhao uses tanh
        h = torch.relu(self.grat2(g, h))
        h = self.grat3(g, h)
        return h


class GRATV3(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, *args):
        super(GRATV3, self).__init__()
        self.grat = GRATV3_(in_feats, hidden_feats1, hidden_feats2, 1, *args)  # out_feats=1

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h


class GRATV4_(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3, out_feats, *args):
        super(GRATV4_, self).__init__()
        self.grat1 = GRATVLayer(in_feats, hidden_feats1)
        self.grat2 = GRATVLayer(hidden_feats1, hidden_feats2)
        self.grat3 = GRATVLayer(hidden_feats2, hidden_feats3)
        self.grat4 = GRATVLayer(hidden_feats3, out_feats)

    def forward(self, g, feature):
        h = torch.relu(self.grat1(g, feature))  # Guanhao uses tanh
        h = torch.relu(self.grat2(g, h))
        h = torch.relu(self.grat3(g, h))
        h = self.grat4(g, h)
        return h


class GRATV4(nn.Module):
    def __init__(self, in_feats, hidden_feats1, hidden_feats2, hidden_feats3, *args):
        super(GRATV4, self).__init__()
        self.grat = GRATV4_(in_feats, hidden_feats1, hidden_feats2, hidden_feats3, 1, *args)

    def forward(self, g, feature):
        h = self.grat(g, feature)
        h = torch.sigmoid(h)
        return h

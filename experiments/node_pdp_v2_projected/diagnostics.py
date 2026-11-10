"""Restricted diagnostics, not release metadata; no real data collected here."""
import torch


def slot_summary(batch,slots):
    if len(batch)!=4 or any(len(s)!=4 for s in batch) or tuple(slots.shape)!=(4,2340):
        raise ValueError('fixed four slots/four positions/layout required')
    counts=[sum(item is not None for item in s) for s in batch]
    return dict(positions=16,effective_positions=sum(counts),empty_candidate_slots=sum(x==0 for x in counts),
                zero_gradient_slots=sum(not bool(torch.count_nonzero(s)) for s in slots))


def definitions():
    return dict(version='node-pdp-v2-projected-diagnostics-v1',scope='local restricted, not DP release',
        margin='pre-sigmoid logit minus last fc bias, on original fixed test nodes',
        signed_counts='negative/zero/positive exact margins on nodes with original outdegree>0; numerical residual separately reported',
        positive_fraction='positive margins / nonzero-outdegree nodes; NA for zero denominator',
        quantiles='min,q25,median,q75,max; zero/nonzero-outdegree groups separately; NA for empty groups',
        zero_aggregate_fraction='nodes whose final aggregate vector is exactly zero / nodes',
        projected_zero_weights='exact zero coordinates among32 final fc weights after projection',
        projection_trigger='at least one processed target weight<0 before projection; per-step and coordinate counts',
        projection_displacement='joint L2 of32 coordinates after minus before projection',
        ties='exact pre-sigmoid equality; largest group, top-k boundary size and boundary crossing',
        zero_outdegree_seed_fraction='selected nodes with original outdegree0 / selected count',
        influence='original directed outgoing one-step IC p=1; k=50; Email reference432 only in separately approved real study',
        actual_zero_gradient_slots='joint slot contribution exactly zero after slot clipping; NOT inferred from candidates',
        output_policy='none of these diagnostics enters privacy-mode release whitelist',
        development_privacy='fixed v2 theorem does not cover historical private-test-driven candidate selection')

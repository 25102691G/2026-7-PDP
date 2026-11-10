"""Read-only, restricted research diagnostics; never a publication mechanism.

Observe existing Python call/return events without wrapping samplers, replacing
random streams, changing gradients, or adding a forward/backward pass. Only
whitelisted frames contribute scalar counters; Gaussian/Bernoulli frames and
their values are never inspected or stored. CPU timing includes profiler cost.
"""
from collections import Counter
import math
import sys


class TrainingObserver:
    def __init__(self):
        from experiments import sampling
        from experiments.node_dp_slots import adapter, aggregation, mechanism
        self.codes = dict(cache=adapter.SlotCache.__init__.__code__, batch=adapter.Adapter.batch.__code__,
                          stage=sampling._frequency_stage.__code__, starts=sampling._candidate_starts.__code__,
                          walk=sampling._walk_from_start.__code__, clip=aggregation.clip.__code__)
        self.center_code = mechanism.center.__code__
        self.double_code = aggregation.double_clip.__code__
        self._wanted = set(self.codes.values())
        self._sampling = {}
        self._item_stage = {}
        self._slot = self._stage = None
        self._stage_number = 0
        self._steps = []

    def __enter__(self):
        if sys.getprofile() is not None:
            raise RuntimeError('research observer cannot replace an existing profiler')
        sys.setprofile(self._observe)
        return self

    def __exit__(self, *_):
        sys.setprofile(None)

    def _observe(self, frame, event, returned):
        code = frame.f_code
        if code not in self._wanted or event not in ('call', 'return'):
            return
        local = frame.f_locals
        if code is self.codes['cache']:
            if event == 'call':
                self._slot = local['j']; self._stage_number = 0
                self._sampling[self._slot] = dict(slot=self._slot, stages=[])
            else:
                if 'result' not in local or not hasattr(local['self'], 'subgraphs'):
                    self._sampling[self._slot]['incomplete_cache'] = True
                    self._slot = self._stage = None
                    return  # Preserve the original resource/backend exception.
                result, cache = local['result'], local['self']
                record = self._sampling[self._slot]
                record.update(candidates=len(cache.subgraphs), scs_candidates=result.stage1_count,
                              bes_candidates=result.stage2_count,
                              candidate_size_counts=dict(Counter(len(item.nodes) for item in cache.subgraphs)))
                record['rwr_attempts'] = sum(s['rwr_attempts'] for s in record['stages'])
                record['rwr_successes'] = sum(s['rwr_successes'] for s in record['stages'])
                for index, item in enumerate(cache.subgraphs):
                    self._item_stage[id(item)] = 'scs' if index < result.stage1_count else 'bes'
                self._slot = self._stage = None
        elif code is self.codes['stage'] and self._slot is not None:
            if event == 'call':
                name = ('scs', 'bes')[self._stage_number]; self._stage_number += 1
                self._stage = dict(stage=name, target_size=local['size'], candidate_starts=0,
                                   rwr_attempts=0, rwr_successes=0)
                self._sampling[self._slot]['stages'].append(self._stage)
            else:
                self._stage = None
        elif code is self.codes['starts'] and event == 'return' and self._stage is not None:
            self._stage['candidate_starts'] += len(returned)
        elif code is self.codes['walk'] and self._stage is not None:
            if event == 'call': self._stage['rwr_attempts'] += 1
            elif returned is not None: self._stage['rwr_successes'] += 1
        elif code is self.codes['batch'] and event == 'return':
            slots = []
            for j, items in enumerate(returned):
                stage_counts = Counter(self._item_stage[id(item)] for item in items if item is not None)
                effective = sum(item is not None for item in items)
                slots.append(dict(slot=j, effective_positions=effective, zero_padding_positions=4-effective,
                                  selected_scs_positions=stage_counts['scs'], selected_bes_positions=stage_counts['bes'],
                                  subgraph_gradients_seen=0, subgraph_norm_above_C_count=0,
                                  subgraph_preclip_norm_max=None, subgraph_postclip_norm_max=None,
                                  slot_preclip_norm=None, slot_postclip_norm=None, slot_norm_above_C=None))
            self._steps.append(dict(iteration=len(self._steps)+1, slots=slots))
        elif code is self.codes['clip'] and event == 'return' and self._steps:
            parent = frame.f_back
            if parent.f_code is self.center_code:
                kind, j = 'subgraph', parent.f_locals['j']
            elif (parent.f_code.co_name == '<listcomp>' and parent.f_back is not None
                  and parent.f_back.f_code is self.double_code):
                kind, j = 'slot', parent.f_locals['j']
            else:
                return
            before = local.get('length')
            before = before if before is not None and math.isfinite(before) else None
            after = float(returned.norm())  # No gradient computation or mutation.
            record = self._steps[-1]['slots'][j]
            if kind == 'slot':
                record.update(slot_preclip_norm=before, slot_postclip_norm=after,
                              slot_norm_above_C=None if before is None else before > 3.)
            else:
                record['subgraph_gradients_seen'] += 1
                record['subgraph_norm_above_C_count'] += int(before is not None and before > 3.)
                for name, value in [('subgraph_preclip_norm_max', before), ('subgraph_postclip_norm_max', after)]:
                    if value is not None: record[name] = max(value, record[name] or 0.)

    def summary(self):
        return dict(scope='restricted diagnostics; not DP output',
                    sampling_definition='RWR is shared by SCS and BES, not a third disjoint candidate source',
                    clipping_trigger_definition='pre-clip joint L2 norm strictly greater than C=3; not numerical epsilon shrinkage',
                    sampling_slots=[self._sampling[j] for j in sorted(self._sampling)], steps=self._steps,
                    secret_random_values_recorded=False, extra_forward_backward=False, timing_includes_observer=True)

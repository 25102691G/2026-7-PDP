"""G2-only paired research: one in-memory outer graph, two fresh Gaussian runs.

No new training loop or privacy publication interface.  All optimization uses
node_dp_slots.training.execute and the unchanged v2 policy/SecretGaussian.
Private cache/batch signatures exist only in memory and are never serialized.
"""
from dataclasses import dataclass
import math
from pathlib import Path

DATASETS=('bitcoin','lastfm','facebook')


def task_matrix():
    return [dict(id=f'{d}_i{i}_h{h}_z{z}',dataset=d,model_seed=i,outer_index=h,gaussian_replicate=z,
                 status='NOT_ATTEMPTED') for d in DATASETS for i in (1,2) for h in (0,1) for z in (0,1)]


def variance_components(groups):
    if len(groups)!=2 or any(len(g)!=2 for g in groups):raise ValueError('fixed2x2 nested design required')
    means=[sum(g)/2 for g in groups];grand=sum(means)/2
    residual=sum((v-m)**2 for g,m in zip(groups,means) for v in g)/2
    between=2*sum((m-grand)**2 for m in means)
    raw=(between-residual)/2
    return dict(gaussian_residual_variance=residual,outer_mean_square=between,outer_variance_raw=raw,
                outer_variance_nonnegative_proxy=max(0.,raw),outer_means=means,
                outer_mean_difference=means[1]-means[0],grand_mean=grand,
                residual_df=2,outer_df=1,interpretation='exploratory nested MOM; raw negative estimate retained')


def _runtime():
    from experiments.node_dp_slots import mechanism as base
    from experiments.node_dp_slots.training import execute
    from experiments.node_dp_slots.noise import SecretGaussian,TestGaussian
    from experiments.node_pdp import sampling,accounting
    from experiments.node_pdp_v2_projected import policy,diagnostics
    return base,execute,SecretGaussian,TestGaussian,sampling,accounting,policy,diagnostics


def _observer(reference):
    from experiments.pdp_research_observer import TrainingObserver
    from experiments.node_dp_slots.adapter import Adapter

    class PairObserver(TrainingObserver):
        def __init__(self):
            super().__init__();self._cache_signature=None;self._batch_signatures=[];self.matched_steps=0

        def _observe(self,frame,event,returned):
            super()._observe(frame,event,returned)
            if event!='return':return
            if frame.f_code is Adapter.__init__.__code__:
                self._cache_signature=frame.f_locals['self'].fingerprints()
                if reference is not None and self._cache_signature!=reference._cache_signature:
                    raise RuntimeError('paired local caches differ')
            elif frame.f_code is Adapter.batch.__code__:
                signature=tuple(tuple(slot) for slot in returned)
                if reference is None:self._batch_signatures.append(signature)
                else:
                    if self.matched_steps>=len(reference._batch_signatures) or signature!=reference._batch_signatures[self.matched_steps]:
                        raise RuntimeError('paired public batch tapes differ')
                    self.matched_steps+=1
    return PairObserver()


@dataclass(repr=False)
class Trajectory:
    model: object
    parameters: tuple
    initial_hash: str
    diagnostics: dict
    sampling_diagnostics: dict


class ControlledPair:
    def __init__(self,registry,spec,graph,*,model_seed):
        *_,sampling,accounting,policy,diagnostics=_runtime()
        self._setup(registry,spec,graph,model_seed)
        self.purpose='research';self.__sampler=sampling.SecretBernoulli()
        self.__held=sampling.sample_once(graph,registry,self.__sampler).graph
        self.__noise_factory=_runtime()[2]

    def _setup(self,registry,spec,graph,seed):
        base,execute,Secret,Test,sampling,accounting,policy,diagnostics=_runtime()
        accounting.validate_public(registry,spec);sampling.validate_graph(graph,spec)
        if type(seed) is not int or seed<0:raise ValueError('public model initialization seed required')
        self.registry=registry;self.spec=spec;self.model_seed=seed;self.__attempts=0
        self.__reference=None;self.__previous_noise=None;self.__initial_hash=None

    @classmethod
    def synthetic(cls,registry,spec,graph,*,model_seed,sampling_seed,noise_seeds):
        if len(noise_seeds)!=2:raise ValueError('two artificial Gaussian seeds required')
        base,execute,Secret,Test,sampling,accounting,policy,diagnostics=_runtime()
        obj=object.__new__(cls);obj._setup(registry,spec,graph,model_seed);obj.purpose='synthetic'
        obj.__sampler=sampling.TestBernoulli(sampling_seed)
        obj.__held=sampling.sample_once(graph,registry,obj.__sampler).graph
        seeds=iter(noise_seeds);obj.__noise_factory=lambda:Test(next(seeds))
        return obj

    def train_next(self):
        base,execute,Secret,Test,sampling,accounting,policy,diagnostics=_runtime()
        if self.__attempts>=2:raise PermissionError('two trajectory attempts consumed; no retry')
        self.__attempts+=1  # Failures consume this trajectory position.
        noise=self.__noise_factory()
        if self.purpose=='research' and (type(self.__sampler) is not sampling.SecretBernoulli or type(noise) is not Secret):
            raise PermissionError('test random sources forbidden in real G2')
        if noise is self.__previous_noise:raise PermissionError('Gaussian source must be a fresh independent instance')
        distinct=noise is not self.__previous_noise;self.__previous_noise=noise
        model=base.new_model(self.model_seed);layout=policy.Layout.from_model(model)
        from experiments.pdp_g0g1_diagnostic import _state_hash
        initial_hash=_state_hash(model)
        if self.__initial_hash is None:self.__initial_hash=initial_hash
        elif initial_hash!=self.__initial_hash:raise RuntimeError('paired initial states differ')
        observer=_observer(self.__reference);model.train()
        with observer:
            values,frames,_=execute(self.spec,self.__held,noise,model=model,
                step=lambda m,s,z:policy.step(m,s,z,layout),
                collector=lambda batch,slots,stats:dict(diagnostics.slot_summary(batch,slots),**stats),
                check_slot_bounds=True,check_noise=True)
        if not all(math.isfinite(v) for v in values):raise RuntimeError('nonfinite model')
        if noise.calls!=100 or len(frames)!=100:raise RuntimeError('fixed100 updates/noise calls required')
        if observer._cache_signature is None:raise RuntimeError('actual cache comparison witness absent')
        second=self.__reference is not None
        if not second and len(observer._batch_signatures)!=100:raise RuntimeError('first100 actual batches not observed')
        if second and observer.matched_steps!=100:raise RuntimeError('paired100 batches not verified')
        if not second:self.__reference=observer
        info=dict(updates=100,noise_calls=noise.calls,noise_coordinates=2340*noise.calls,outer_graph_draws_per_pair=1,
                  shared_outer_graph_in_memory=True,paired_initial_states_equal=True if second else None,
                  paired_caches_equal=True if second else None,paired_batch_steps_equal=observer.matched_steps if second else None,
                  distinct_noise_source=distinct,projection_calls=100,
                  all_projected_weights_nonnegative=all(f['nonnegative'] for f in frames),all_parameters_finite=True,
                  steps=frames,publication_allowed=False,mode=self.purpose)
        return Trajectory(model,values,initial_hash,info,observer.summary())

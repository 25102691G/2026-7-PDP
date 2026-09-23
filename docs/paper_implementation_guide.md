# PrivIM paper implementation guide

This document is a durable navigation aid for implementing
`PrivIM_Report.pdf`, titled *PrivIM: Differentially Private Graph Neural
Networks for Influence Maximization*. The PDF remains authoritative whenever
this guide and the paper differ.

## Method-name mapping

| Paper terminology | Meaning | Repository setting |
|---|---|---|
| PrivIM | Algorithm 1 theta-bounded RWR plus Algorithm 2 DP training | `PrivIM` |
| PrivIM + SCS | Algorithm 3 Stage 1 only | `PrivIM+SCS` |
| PrivIM* / PrivIM + SCS + BES | Full Algorithm 3, both stages | `PrivIM+` (preferred compatibility name) or `PrivIM*` |
| Non-Private | Full PrivIM* with `epsilon = infinity` | `PrivIM+ --no-dp`; add `--clip-non-private` to retain Algorithm 2 clipping with zero noise |

`PrivIM+` is a repository compatibility alias, not the name used for the full
method in the paper. User-facing scientific results should label it
`PrivIM* (SCS+BES)`.

## Paper-to-code map

| Paper component | Primary implementation |
|---|---|
| Algorithm 1, theta projection and fixed-size r-hop RWR | `experiments/sampling.py`, `experiments/load_graph.py` |
| Algorithm 2, per-subgraph clipping and DP-SGD | `experiments/train.py`, `experiments/privacy.py` |
| Equation 5, probabilistic penalty loss | `model/functional/MaxCoverLoss.py` |
| Equations 6-8, occurrence bound, sensitivity, RDP accounting | `experiments/train_models.py`, `experiments/privacy.py` |
| Algorithm 3, SCS and BES | `experiments/sampling.py`, `experiments/load_graph.py` |
| Equations 10-12 and Appendix H parameter indicator | `experiments/config.py` |
| Paper experiment launcher | `experiments/run_reproduction.py` |
| Influence spread and CELF coverage ratio | `experiments/evaluate_models.py` |

## Algorithmic invariants

### Input feature protocol

The paper defines a node feature matrix `X` but does not publish how `X` is
constructed in the experiments. The released implementation uses a 32-D
all-one vector, exposed as the default `--feature-protocol ones`. The initial
`degree` diagnostic used the 3-D vector `[1, normalized in-degree, normalized
out-degree]`, but changing the input width also changed RNG consumption and
therefore every later layer's initialization. Its Facebook result
`87.72 +/- 9.07` is exploratory rather than a controlled feature attribution.
The corrected `degree32` protocol places the same three values in the first
three coordinates and pads the other 29 coordinates with zero. It preserves
the released model's parameter shapes and gives a paired, identical model
initialization for each seed. In controlled five-seed non-private runs it
improved Facebook from `84.55 +/- 11.65` to `91.45 +/- 6.32` and Email from
`96.85 +/- 1.58` to `97.73 +/- 0.93`; the paper reports `97.50 +/- 0.34` and
`98.09 +/- 1.69`, respectively. Additional degree32 results are Bitcoin
`98.81 +/- 0.89` (paper 98.99) and LastFM `94.44 +/- 2.74` (paper 97.93).
However, paired all-one controls reached `99.47 +/- 0.72` on Bitcoin and
`97.48 +/- 2.13` on LastFM, outperforming degree32 on both. Degree32 is thus
not a universal replacement for the released feature convention. The final
paired HepPh/Gowalla runs reinforce this conclusion: degree32 produced
`79.79 +/- 15.46` and `87.59 +/- 13.93`, while all-one controls produced
`72.45 +/- 11.06` and `99.22 +/- 1.33`, respectively. Degree32 modestly raises
the unstable HepPh mean but severely degrades Gowalla; neither HepPh result
approaches the paper's `98.42 +/- 0.31`. The six-dataset diagnostic is now
complete, `ones` remains the uniform reproduction default, and dataset-specific
post-hoc feature selection must not be presented as one uniform protocol.
Degree normalization is computed on the complete prepared message graph
before subgraph induction. Both degree protocols remain repository
diagnostics, not values claimed by the paper. Training, evaluation, checkpoint
metadata and resume validation must use the same feature protocol and input
dimension.

### Equation (5) loss penalty

The paper defines a positive trade-off coefficient `lambda` in Equation (5)
but does not report its experimental value. Revision `31166ca` hard-codes
`lambda=1`, retained as the uniform default. The explicit
`--loss-penalty` diagnostic is recorded as `loss_lambda` and participates in
resume validation. On HepPh, `lambda=0.5` raises Non-Private from
`73.76 +/- 9.20` to `92.12 +/- 7.98` and strict epsilon 4 PrivIM* from
`61.63 +/- 23.63` to `90.68 +/- 6.15`; epsilon 1 remains unstable at
`48.04 +/- 31.45`. This is evidence that the unpublished coefficient is a
material Table-II ambiguity, but post-hoc dataset-specific selection must not
be presented as a uniform paper setting.

### Algorithm 1: base PrivIM

- Project the input graph to maximum in-degree `theta` before sampling.
- Select each potential starting node at rate `q`.
- Run RWR with return probability `tau`, constrained to the starting node's
  `r`-hop neighborhood.
- Emit a subgraph only after collecting exactly `n` unique nodes within walk
  length `L`.
- For an `r`-layer GNN, the paper bounds maximum node occurrence by
  `Ng = sum(theta**i for i in 0..r) = (theta**(r+1)-1)/(theta-1)`.

### Algorithm 2: private training

- A single subgraph is one DP sample.
- Uniformly sample a batch of `B` subgraphs from the container.
- Compute and clip each subgraph gradient separately to L2 norm `C`.
- Sum clipped gradients, add calibrated Gaussian noise using sensitivity
  `Delta_g <= C * Ng`, then divide by the batch size for the update.
- Theorem 3 prints a binomial mixture, while Algorithm 2 describes a uniformly
  sampled fixed-size batch. The repository exposes both interpretations as
  `paper_binomial` and `fixed_batch_hypergeometric`; results must name which
  one was used.
- Compose the selected RDP bound over `T` iterations and convert it to
  `(epsilon, delta)`-DP using Theorem 1.
- The privacy adjacency relation is unbounded node-level adjacency.

The paper defines Non-Private as PrivIM* with `epsilon = infinity` (Section
V-A), but does not explicitly state whether its implementation removes only
the Gaussian perturbation or the whole DP-SGD transform.  The limiting form
of Algorithm 2 keeps Lines 4--7 (per-subgraph clipping and averaging) and sets
the Line 8 noise to zero.  The repository therefore exposes that
interpretation as `--clip-non-private`; the compatibility default remains
ordinary batch-mean SGD until the full five-run baseline has been validated.
Checkpoint metadata distinguishes the two interpretations with
`non_private_clipping` and `gradient_reduction`.

### Algorithm 3: enhanced sampling

- SCS does not theta-project the graph.
- Maintain a global node-frequency vector `f`; never allow a node frequency
  above threshold `M`.
- Among eligible neighbors, sampling weight is proportional to
  `1 / (f[v] + 1)**mu`; nodes at `M` have zero weight.
- SCS yields the tighter occurrence bound `Ng* = M`.
- BES removes nodes that reached `M`, rebuilds the remaining graph, reduces
  subgraph size to `n/s`, and runs frequency sampling again with the same
  global threshold. It supplements boundary structure without increasing the
  occurrence bound or privacy budget.

## Paper experiment protocol

Unless a task explicitly changes the protocol, reproduction runs should use:

- six datasets: Email, Bitcoin, LastFM, HepPh, Facebook, and Gowalla;
- random 50/50 train/test node split;
- IC evaluation with edge influence probability `1`, one diffusion step, and
  seed-set size `k=50`;
- privacy budgets `epsilon` from 1 through 6 and
  `delta < 1 / |V_train|`;
- `q = 256 / |V_train|`, `L=200`, `theta=10`, `tau=0.3`;
- learning rate `0.005`; the reproduction default keeps it fixed
  (`gamma=1`). Algorithm 2 denotes a general `eta_t`, while the released code
  defaults to `StepLR(step_size=5, gamma=0.99)`. A controlled Facebook run
  with that decay produced `86.41 +/- 1.12`, below the fixed-rate
  `88.69 +/- 5.16`, so the scheduler remains an explicit compatibility option
  rather than the paper default;
- `T=500` optimizer updates in this repository's validated reproduction
  protocol; Algorithm 2 names `T` but the paper does not publish its numeric
  value, so this is an explicit empirical choice rather than a paper constant;
- three-layer GRAT with 32 hidden units per layer;
- five independent runs, reporting mean and dispersion;
- effectiveness as influence spread and coverage ratio
  `spread(method) / spread(CELF)`;
- efficiency measured separately for preprocessing and per-epoch training.

Evaluation ranks nodes by pre-sigmoid logits. This preserves the mathematical
ordering of the probabilities specified in Section III-C while avoiding
floating-point ties after sigmoid saturation. Equal logits use node ID as a
deterministic secondary key, and result metadata includes the saturation
fraction and top-k boundary ambiguity. This makes tied runs reproducible but
does not treat genuine equal model scores as learned signal. Optional EMA
parameter averaging is exposed only as a strict-DP post-processing diagnostic
and is not part of the paper-default protocol. Batch size and `T` must remain explicit,
and every tested combination must recalibrate noise using its actual optimizer
step count.

The initial controlled Email PrivIM* diagnostic at epsilon 4 and `T=200`
reduced sample standard deviation from `35.30` for the final iterate to `27.05`
with `B=16, EMA=0.99`, while mean coverage decreased from `56.67` to `54.44`.
Using `B=8` with the same EMA gave `55.28 +/- 26.64` and increased
post-average noise. EMA therefore remains a stability diagnostic, not a
validated Table-II remedy or default.

SGD momentum is likewise an explicit post-privatization diagnostic rather
than a paper default. The controlled `epsilon=4`, `T=200`, `B=16` comparison
with momentum `0.9` changed PrivIM* from `56.67 +/- 35.30` to
`41.20 +/- 25.41` on Email, `91.17 +/- 3.89` to `71.75 +/- 31.62` on
Facebook, and `98.48 +/- 1.03` to `80.01 +/- 27.16` on Gowalla. Keep the
strict default at zero.

Increasing the strict batch size from 16 to 32 at `epsilon=4`, `T=200` also
failed as a uniform remedy. PrivIM* changed from `56.67 +/- 35.30` to
`53.61 +/- 39.55` on Email, from `91.17 +/- 3.89` to `87.27 +/- 8.91` on
Facebook, and from `98.48 +/- 1.03` to `98.51 +/- 1.06` on Gowalla.

### Released-code privacy compatibility warning

The released training loop is not a valid implementation of the paper's
stated privacy composition. It sets `n_epoch=20` and calibrates Gaussian noise
for 20 steps, but each epoch traverses every DataLoader batch. The actual
number of optimizer updates is therefore
`20 * ceil(|Gsub| / B)`, not 20. With the containers produced by the current
paper sampler this ranges from 40 updates on Email PrivIM* to 340 updates on
Facebook PrivIM*. Re-accounting the released noise over the actual updates
under the released accountant's normalization raises epsilon=4 to
approximately 6.05 and 15.70, respectively. The same code also multiplies the
noise occurrence multiplier by 2, inherited from a bounded/replacement-style
sensitivity implementation, whereas the paper specifies unbounded adjacency
and Equation 7 gives `C * Ng`. If that factor is treated as additional noise
under the paper's unbounded sensitivity, the corresponding diagnostics are
approximately 2.53 and 6.49. Neither interpretation repairs the mismatch:
the released loop did not compose its actual update count, and its sensitivity
convention conflicts with Equation 7.

The released base-PrivIM path also uses the manually supplied
`score_upper_bound=10` and sampling hop `r=1`, despite the paper specifying a
three-layer GRAT and Equation 6 defining
`Ng = sum(theta**i, i=0..r)`. The paper-faithful values `theta=10, r=3` give
`Ng=1111`; for the observed base containers `Ng >= |Gsub|`, so every subgraph
may be affected and there is no privacy amplification. This produces much
larger noise than the released path. Consequently, Table II utility cannot be
treated as evidence that the released checkpoints satisfy the paper's stated
epsilon without resolving these inconsistencies. The repository keeps actual
optimizer updates, `Ng`, and achieved epsilon explicit rather than silently
emulating the under-accounted path.

The launcher exposes this training/privacy behavior only as
`--protocol-profile released-compatibility`. It restores 20 full shuffled
DataLoader epochs, 20 calibration steps, private-SGD momentum 0.9,
`StepLR(step_size=5, gamma=0.99)`, base occurrence bound 10, the
fixed-batch accountant, the released noise factor 2, fixed model/batch/noise
streams, and a continuous sampling RNG across rounds. Metadata sets
`strict_privacy_claim=false` and stores both actual and accounted steps plus
both epsilon diagnostics. A one-round Email PrivIM* smoke run produced 40
actual updates and 98.84% coverage; it is evidence about released-code
provenance, not a valid Table II reproduction claim at strict epsilon 4.
The profile defaults to the current tested Algorithms 1/3 samplers and
prepared data rather than silently reviving legacy sampling code.
In particular, it keeps sampling at `r=3`: applying the released `r=1`
literally to Algorithm 1 with `theta=10` gives at most 11 vertices in the
bounded one-hop neighborhood and cannot form `n=40` subgraphs. The released
manual bound 10 remains recorded separately as the compatibility conflict.

The launcher also exposes the released sampler separately as
`--sampling-protocol released-legacy`; it is never selected implicitly and
requires `--protocol-profile released-compatibility`. Revision `31166ca` did
not implement Algorithms 1/3 literally: it repeatedly chose starting vertices
until collecting approximately `q|V|` successful subgraphs, considered only
the first 1,000 eligible vertex IDs at each retry, traversed all incident edges
even for directed graphs, omitted the starting vertex from the sampled node
set, and reused the original graph for a BES stage capped at
`ceil(q|V|/4)` half-sized subgraphs. The same frequency-threshold sampler was
used for its setting named `PrivIM`, without Algorithm 1's theta projection or
r-hop constraint. Checkpoint metadata identifies this diagnostic with
`sampling_protocol=released-legacy`, records the ignored configured hop count,
neighbor mode, target counts, and candidate cap, and continues to set
`strict_privacy_claim=false`. The default `sampling_protocol=paper` remains
the tested implementation of paper Algorithms 1 and 3.

The completed six-dataset `released-compatibility` matrix using the paper
sampler has aggregate Table-II MAE 9.40 percentage points. Its largest gaps
are Facebook PrivIM* (`59.80 +/- 6.89`, paper `95.55`), Facebook PrivIM
(`64.31 +/- 33.57`, paper `82.60`), LastFM PrivIM*
(`76.78 +/- 10.56`, paper `93.76`), and Gowalla SCS
(`78.65 +/- 8.26`, paper `95.47`). In Facebook, the paper BES implementation
increased the container from roughly 115 SCS subgraphs to roughly 270 total
subgraphs. Under the released 20-epoch loop this changed approximately 150
optimizer updates into 350 while calibration remained fixed at 20 steps.
This motivates the isolated released-sampler diagnostic; it does not justify
changing the paper sampler or making a strict epsilon claim.

The targeted released-sampler matrix confirms that this historical behavior
is not a uniform Table-II solution. On Facebook it changes PrivIM* from
`59.80 +/- 6.89` to `88.69 +/- 2.62`; on LastFM and Gowalla it changes SCS
from `79.50 +/- 7.82` and `78.65 +/- 8.26` to `93.27 +/- 3.46` and
`94.02 +/- 3.67`. However, the same sampler collapses base PrivIM to 24.00,
36.11, and 5.68 on Facebook, LastFM, and Gowalla because it is not paper
Algorithm 1. BES also lowers legacy-sampler SCS from 93.27 to 80.50 on LastFM
and from 94.02 to 83.04 on Gowalla. In those runs BES changes both the
container and the number of full-epoch updates, so the launcher exposes
`--compatibility-update-cap N` to isolate the latter. This cap is accepted
only with `released-compatibility`; it truncates actual updates while keeping
the released 20-step calibration unchanged and remains non-strict.

The paired cap results are Facebook PrivIM* `85.27 +/- 4.63` at 320 updates,
LastFM PrivIM* `90.48 +/- 5.72` at 120 updates, and Gowalla PrivIM*
`92.53 +/- 5.57` at 320 updates. Compared with the uncapped runs, the cap
hurts Facebook by 3.42 points but improves LastFM and Gowalla by 9.98 and 9.49
points. Thus container-dependent update count materially changes utility, but
dataset-specific post-hoc caps are not a valid uniform reproduction protocol.

Revision `31166ca` also generated noise with direct `torch.randn_like` calls,
whereas the corrected implementation uses a dedicated generator seeded by
`dp_noise_seed`. The launcher exposes this isolated difference as
`--dp-noise-stream independent|legacy-global`. The latter is compatibility
only, does not apply the dedicated DP-noise seed, and cannot carry a strict
privacy/reproduction claim. Algorithm 2 specifies Gaussian perturbation but
does not prescribe an RNG implementation, so `legacy-global` is provenance
emulation rather than a paper algorithm requirement.

The paper's baseline set is PrivIM, PrivIM*, EGN, HP, HP-GRAT, Non-Private,
and CELF. The repository does not currently implement every external baseline;
do not relabel a local method as an unavailable competitor.

## Reference checkpoints from Table II

These values are regression targets for the experimental setup, not unit-test
constants. Coverage ratios are percentages.

| epsilon | Method | Email | Bitcoin | LastFM | HepPh | Facebook | Gowalla |
|---:|---|---:|---:|---:|---:|---:|---:|
| infinity | Non-Private | 98.09 | 98.99 | 97.93 | 98.42 | 97.50 | 99.78 |
| 4 | PrivIM | 76.08 | 73.48 | 79.39 | 73.54 | 82.60 | 82.26 |
| 4 | PrivIM+SCS | 95.08 | 92.24 | 89.56 | 93.91 | 90.92 | 95.47 |
| 4 | PrivIM* | 94.44 | 92.84 | 93.76 | 94.50 | 95.55 | 98.02 |
| 1 | PrivIM | 62.01 | 60.13 | 31.72 | 32.24 | 70.20 | 68.05 |
| 1 | PrivIM+SCS | 83.05 | 85.76 | 80.72 | 80.98 | 82.68 | 81.28 |
| 1 | PrivIM* | 83.87 | 86.42 | 83.50 | 85.49 | 85.10 | 86.57 |

Large deviations from these targets require investigation of dataset version
and orientation, split semantics, sampling, privacy calibration, training,
and evaluation before drawing scientific conclusions.

## Workflow for future changes

1. Locate the relevant paper section and record its algorithm/equation number.
2. Trace the corresponding files using the map above.
3. State whether the change is exact reproduction, compatibility work, an
   ablation, or an intentional extension.
4. Add or update tests for mathematical and sampling invariants.
5. Run focused tests, then the full test suite when the environment permits.
6. For changes affecting reported results, run a small fixed-seed smoke test
   before launching the full six-dataset matrix.
7. Report the paper reference, tests, observed deviations, and any unresolved
   ambiguity in the final handoff.

Full launcher runs are manifest-scoped below `outputs/runs/<run_id>`. Do not
compare or aggregate checkpoints across manifests unless the differing
protocol fields have been reviewed explicitly.

## Known interpretation cautions

- The paper uses `r` both for GNN depth/hop dependence; keep the configured
  GNN depth and sampling neighborhood consistent when applying Equation 6.
- Coverage ratio uses CELF influence spread as its denominator; it is not the
  fraction of all graph nodes influenced.
- The paper reports means over five randomized runs. A result with high
  between-seed variance should not be characterized using only its mean.
- Randomness is separated into sampling, model, batch, and DP Gaussian-noise
  seeds. The default
  diagnostic protocol fixes the sampling seed across five rounds and varies
  model/batch/noise seeds, so no round reuses those streams accidentally.
  Diagnostic runs can fix any three streams and vary the fourth. `per-round`
  sampling is retained for a separate variability measurement; fixing model,
  batch, and DP-noise seeds in that mode isolates sampling alone. Every
  checkpoint records the realized seeds and the SHA-256 of its ordered sampled
  node sets. The compatibility-only `legacy-global` noise stream deliberately
  consumes PyTorch's global device RNG and records that `dp_noise_seed` was not
  applied.
- Do not assume that increasing epsilon alone explains a result if the sampled
  subgraph container also changes across runs; separate sampling,
  initialization, and DP-noise randomness during diagnosis.
- A 50-update run is a smoke test, not an effectiveness reproduction. It left
  the non-private model strongly initialization-dependent; 500 updates moved
  the five-run Email result from 46.48 +/- 41.20 to 96.85 +/- 1.58.
- Keep the learning-rate schedule in result metadata when comparing runs.
  The released `gamma=0.99` schedule reduces Facebook run-to-run dispersion,
  but the controlled result above shows that it does not close the paper gap.
- `legacy_released` remains the Table II reproduction default for GRAT.
  `legacy_leaky_attention` is an isolation diagnostic that preserves the
  released parameterization and changes only its edge-score activation from
  ReLU to the LeakyReLU specified by Appendix G, Equation 40. On the
  controlled Facebook five-seed baseline it reduced coverage from
  `84.55 +/- 11.65` to `73.80 +/- 21.83`: two seeds reached 95.82/98.73,
  while three collapsed to 51.66--62.37 with 53%--63% sigmoid saturation.
  This hybrid is therefore diagnostic-only and must not become the default.
  `paper_appendix_g` is an explicit diagnostic variant implementing Equations
  37-40 in their stated order. The two conflict: released code uses all-one
  inputs, ReLU attention on untransformed features, and a biased transform
  after aggregation. On Facebook the paper variant improved
  `88.69 +/- 5.16` to `93.09 +/- 3.07`, but on directed Email it collapsed
  from `96.85 +/- 1.58` to `25.65 +/- 19.27`. With all-one inputs, literal
  successor-normalized Appendix G attention also makes the first-layer
  aggregate identical across nodes. Do not promote the paper variant until
  the missing input-feature/direction convention is resolved.

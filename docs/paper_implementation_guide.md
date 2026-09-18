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
not a universal replacement for the released feature convention, and
dataset-specific post-hoc feature selection must not be presented as one
uniform reproduction protocol.
Degree normalization is computed on the complete prepared message graph
before subgraph induction. Both degree protocols remain repository
diagnostics, not values claimed by the paper, until cross-dataset validation
and paired ones controls are complete. Training, evaluation, checkpoint
metadata and resume validation must use the same feature protocol and input
dimension.

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
- Randomness is separated into sampling, model, and batch seeds. The default
  diagnostic protocol fixes the sampling seed across five rounds and varies
  model/batch seeds, so initialization variance is not confounded with a
  different Algorithm 1/3 container. `per-round` sampling is retained for a
  separate variability measurement; fixing model and batch seeds in that mode
  isolates sampling alone. Every checkpoint records the realized seeds and
  the SHA-256 of its ordered sampled node sets.
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

# PrivIM experiment and implementation handoff

Last updated: 2026-09-21 (Asia/Shanghai)

This document is the durable handoff for continuing the PrivIM reproduction
on another machine. Read it together with `AGENTS.md`,
`PrivIM_Report.pdf`, and `docs/paper_implementation_guide.md`. The PDF remains
the primary technical specification.

## 1. Critical repository state

- Branch: `main`
- Current source/migration commit: `2fe2645`, including implementation commit
  `9e4ec16` (`Non private对齐`) and guide correction `aa338b5`, based on
  `f4a5ea3`. Local `main` and `origin/main` matched before the documentation
  updates recorded in this handoff.
- Paper SHA-256:
  `cf7fc97430366bec111d230d1a945a79c844980dd791373c1d04374514c647ad`
- Current test status: **49 tests passing** with
  `.venv/bin/python -m unittest discover -s tests`.
- The source, documentation, and tests are committed. The deterministic
  prepared splits and a curated set of run manifests/evaluation summaries are
  included by the migration commit described below.
- `outputs/` is about 29 MB, `dataset/prepared/` about 15 MB,
  `raw_dataset/` about 31 GB, and `.venv/` about 6.4 GB on the current machine.
  Recreate `.venv` on the new device rather than copying it.
- Full experimental outputs and raw data remain ignored by Git. The curated
  records do not include model checkpoints, so separately archive
  `outputs/runs/` when exact checkpoint-level continuation is required.
- Raw data are not committed. Verify redownloaded or separately copied files
  against `raw_dataset/checksums.sha256`.

## 2. Tested environment

| Component | Version |
|---|---|
| Python | 3.8.20 (conda-forge, GCC 13.3.0) |
| PyTorch | 2.1.1+cu121 |
| DGL | 2.0.0+cu121 |
| igraph | 0.11.5 |
| NumPy | 1.24.1 |
| pandas | 2.0.3 |

The tool shell used to write this handoff did not expose CUDA, but the user's
interactive terminal successfully ran GPU experiments. Use `--device auto`.

## 3. Paper mapping and fixed scientific protocol

Method names:

| Paper | Repository |
|---|---|
| PrivIM | `--settings PrivIM` |
| PrivIM + SCS | `--settings PrivIM+SCS` |
| PrivIM* (SCS+BES) | `--settings PrivIM+` (compatibility name) |
| Non-Private | `PrivIM+ --no-dp`, stored as `eps_10000` |

Current reproduction protocol:

- unbounded node-level DP;
- generated 50/50 node split, seed 0;
- isolated policy `active` and self-loops removed;
- IC evaluation, edge probability 1, one diffusion hop, seed-set size 50;
- `q = 256 / |V_train|`, walk length 200, restart probability 0.3;
- theta 10, three-layer GRAT, hidden width 32;
- batch size 16, learning rate 0.005, fixed schedule (`gamma=1`);
- 500 optimizer updates; the paper names `T` but does not publish its value;
- five runs;
- diagnostic comparisons fix sampling and batch seeds and vary model seed;
- CELF is the coverage-ratio denominator.

Table II non-private targets:

| Email | Bitcoin | LastFM | HepPh | Facebook | Gowalla |
|---:|---:|---:|---:|---:|---:|
| 98.09 | 98.99 | 97.93 | 98.42 | 97.50 | 99.78 |

## 4. Major implementation work completed

### Reproduction and provenance

- Added manifest-scoped launcher `experiments/run_reproduction.py`.
- Added strict prepared-data preflight and run-local evaluation roots.
- Added `experiments/validate_results.py` for Table II comparison, run-count,
  and stability checks.
- Checkpoint JSON records algorithm, model, optimizer, privacy, data, feature,
  seed, sampler hash, timing, and diagnostic metadata.
- Resume validates iterations, learning rate/schedule, GRAT variant,
  initialization, seed protocol, non-private clipping, and feature protocol.

### Sampling and privacy

- Implemented explicit Algorithm 1 theta-bounded sampling and Algorithm 3
  SCS/BES paths.
- Privacy quantities `n`, theta, `r`, `M`, `Ng`, clipping, noise, batch size,
  sampling and iterations are explicit.
- Implemented per-subgraph DP-SGD and two named accounting interpretations:
  `paper_binomial` (Theorem 3 expression, default) and
  `fixed_batch_hypergeometric` (fixed-size Algorithm 2 interpretation).
- `--clip-non-private` implements the literal zero-noise limit of Algorithm 2
  while keeping per-subgraph clipping. It is diagnostic-only.

### Private reproduction audit

Controlled epsilon=4 experiments separated model, sampling, batch, and DP
noise seeds. Batch variability was negligible, sampling was generally small,
model initialization dominated Facebook variance, and DP noise dominated
Email variance. At T=500, Email PrivIM* remained bimodal and Facebook PrivIM*
reached `90.33 +/- 4.37`; increasing T raises the per-update noise required by
composition and does not remove the unstable trajectories.

The first strict EMA/batch diagnostics on Email PrivIM* at epsilon 4 and
`T=200` are complete. Re-evaluating the original `B=16`, final-iterate
checkpoints with deterministic logit tie-breaking gives the unchanged
`56.67 +/- 35.30`. Holding the same five seeds, accountant, container and
`B=16` while saving an EMA with decay 0.99 gives `54.44 +/- 27.05`; changing
the batch size to 8 within the EMA experiment gives `55.28 +/- 26.64`. Both
runs achieve epsilon about 3.997. EMA reduces dispersion and sigmoid
saturation but does not raise mean utility, and it compresses logits enough to
increase top-k boundary ties. Batch 8 also raises the post-average
per-coordinate noise from 5.55 to 6.38. Keep EMA explicit and diagnostic-only;
do not promote it to the paper default from this result.

Two further strict `epsilon=4`, `T=200` five-seed pilots are complete. Raising
the batch size from 16 to 32 did not improve PrivIM*: Email changed from
`56.67 +/- 35.30` to `53.61 +/- 39.55`, Facebook from `91.17 +/- 3.89` to
`87.27 +/- 8.91`, and Gowalla from `98.48 +/- 1.03` to
`98.51 +/- 1.06`. Applying SGD momentum `0.9` after private-gradient
construction was substantially worse: `41.20 +/- 25.41`,
`71.75 +/- 31.62`, and `80.01 +/- 27.16`, respectively. Keep `B=16` and
momentum zero as strict defaults. The runs are `strict-b32-t200-pilot` and
`strict-momentum09-t200-pilot`.

Evaluation summary grouping now retains Non-Private checkpoints whose DP-only
metadata fields are null. The six-dataset `paper-all-eps1-4-20260922` summary
therefore contains 42 rows: 18 for epsilon 1, 18 for epsilon 4, and six
Non-Private baselines.

The HepPh Oracle/data audit is complete. The raw graph exactly matches paper
Table I (12,008 nodes and 118,489 undirected edges), prepared local IDs are
contiguous and disjoint, and CELF covers 1,741 test nodes at k=50. Evaluating
the same checkpoints on the complete raw graph gives `72.55 +/- 8.41`, close
to the prepared-test result `73.76 +/- 9.20`, so removing cross-split edges is
not the main gap. The default models overlap CELF's Top-50 by only 20.2 nodes
on average.

Equation (5) specifies `lambda > 0` but the paper does not publish its value;
the released code hard-codes 1. An explicit loss-penalty sweep found a strong
HepPh-specific improvement at `lambda=0.5`: Non-Private becomes
`92.12 +/- 7.98`, and strict epsilon 4 PrivIM* becomes
`90.68 +/- 6.15` at accounted epsilon 3.995, versus the prior
`61.63 +/- 23.63`. Strict epsilon 1 improves only from
`41.07 +/- 30.50` to `48.04 +/- 31.45` and retains severe boundary ties, so
noise remains dominant there. `lambda=16`, suggested by an unconstrained
full-test-graph soft oracle, collapses real subgraph training to
`5.63 +/- 0.38` because its scale does not transfer to n=60 training
subgraphs. Keep the uniform default at 1; report 0.5 as a HepPh diagnostic
until the authors' experimental lambda is known. Relevant runs are
`diagnostic-hepph-lambda05`, `diagnostic-hepph-lambda05-eps4`, and
`diagnostic-hepph-lambda05-eps1`.

Parameter growth is quantitatively noise-dominated. Email PrivIM* at T=500
has 2,340 trainable parameters and per-coordinate post-average noise standard
deviation 8.72. With learning rate 0.005, the noise-only parameter displacement
RMS is about `sqrt(2340*500)*0.005*8.72 = 47.1`, while observed final parameter
norms are 46.6--47.9. New checkpoint diagnostics record clipped-gradient,
noise, noisy-gradient norms and the expected cumulative noise displacement.

The released code contains two privacy-critical compatibility conflicts:

1. it calibrates for `n_epoch=20` privacy steps but performs
   `20 * ceil(|Gsub|/16)` optimizer updates (40 for Email PrivIM* and 340 for
   Facebook PrivIM* in the audited containers); re-accounting changes claimed
   released-accountant epsilon=4 to about 6.05 and 15.70 respectively. The
   released optimizer also multiplies its noise occurrence multiplier by 2;
   interpreting that factor as extra noise under the paper's unbounded
   Equation 7 gives separate diagnostics of about 2.53 and 6.49;
2. base PrivIM uses `score_upper_bound=10` and sampling `r=1`, rather than the
   paper Equation 6 value `Ng=1111` for theta 10 and the three-layer model.

These discrepancies explain why strictly accounted runs add much more noise
than the released empirical path. Do not reduce `T`, `Ng`, or the noise merely
to recover Table II without labeling the resulting run as a non-private or
under-accounted compatibility diagnostic.

An explicit `released-compatibility` launcher profile now isolates this path.
It restores the released 20-epoch traversal, 20-step calibration, base
bound 10, momentum 0.9, gamma 0.99 scheduler, fixed-batch accountant, factor-2
noise convention, fixed model/batch/noise streams, and a sampling stream that
advances across rounds. Metadata refuses a strict epsilon claim and records
accounted steps, actual updates, and both epsilon interpretations. The GPU
smoke run `smoke-released-compatibility-v2` completed 40 updates for Email
PrivIM* and obtained 98.84% coverage in one round.
The current Algorithm 1 sampler remains at `r=3`; literal released `r=1` with
`theta=10` cannot collect `n=40` nodes from a bounded one-hop neighborhood.
The released manual base bound 10 is recorded independently as an intentional
compatibility conflict.

The complete compatibility matrix with the paper sampler is stored in
`released-compat-eps4-small`, `released-compat-eps4-medium`, and
`released-compat-eps4-gowalla`. Across all 18 configurations its Table-II MAE
is about 9.40 percentage points and 9 configurations have sample standard
deviation above 5. Notable failures are Facebook PrivIM*
`59.80 +/- 6.89`, LastFM PrivIM* `76.78 +/- 10.56`, and Gowalla SCS
`78.65 +/- 8.26`. Facebook SCS is `91.05 +/- 11.23`, but adding the paper BES
stage expands its sampled container from about 115 to about 270 subgraphs;
the released 20-epoch training loop consequently grows from about 150 to 350
updates while still accounting for 20.

An isolated `--sampling-protocol released-legacy` diagnostic now reproduces
revision `31166ca`'s sampling quirks without changing the paper default. It
retries starts until `q|V|` successful samples, scans at most the first 1,000
eligible IDs, traverses all incident edges on directed graphs, omits the start
from the sampled set, and caps BES at `ceil(q|V|/4)` half-sized subgraphs on
the unreduced original graph. It requires
`--protocol-profile released-compatibility`, is recorded in manifests,
checkpoints, and evaluation summaries, and never carries a strict epsilon
claim. The next bounded experiment should compare this sampler on Facebook,
LastFM, and Gowalla; do not rerun the existing paper-sampler matrix.
The end-to-end GPU smoke `smoke-released-legacy-email-gpu` generated 31 large
and 2 boundary subgraphs for Email PrivIM*, performed 60 actual updates while
accounting for 20, and obtained 92.82% coverage in one round. Its nominal
actual-step epsilon diagnostic is 7.77 (3.21 when the released factor-two
noise is treated as extra noise), so it is provenance evidence only.

The completed targeted released-sampler runs are
`diagnostic-released-legacy-facebook`,
`diagnostic-released-legacy-lastfm`, and
`diagnostic-released-legacy-gowalla`. Their key results are:

| Dataset | Legacy PrivIM | Legacy SCS | Legacy PrivIM* |
|---|---:|---:|---:|
| Facebook | 24.00 +/- 2.65 | 79.96 +/- 13.65 | 88.69 +/- 2.62 |
| LastFM | 36.11 +/- 7.93 | 93.27 +/- 3.46 | 80.50 +/- 9.96 |
| Gowalla | 5.68 +/- 8.03 | 94.02 +/- 3.67 | 83.04 +/- 3.27 |

The legacy sampler therefore helps selected enhanced configurations but is
not a unified reproduction protocol. In particular, BES raises Facebook but
lowers LastFM and Gowalla relative to their legacy SCS runs. It also raises
actual updates from 320 to 400 on Facebook/Gowalla and from 120 to 140--160
on LastFM. `--compatibility-update-cap N` now isolates that update-count
effect without changing the sampled container, scheduler, momentum, or
20-step privacy calibration. It is diagnostic-only and requires the released
compatibility profile.

The completed paired cap experiments give Facebook PrivIM* `85.27 +/- 4.63`
at 320 updates (down from `88.69 +/- 2.62` at 400), LastFM PrivIM*
`90.48 +/- 5.72` at 120 updates (up from `80.50 +/- 9.96` at 140--160), and
Gowalla PrivIM* `92.53 +/- 5.57` at 320 updates (up from `83.04 +/- 3.27` at
400). The sampled-node hashes, initial model norm, and realized seeds match
round-by-round within each pair. Update count is therefore a causal source of
the LastFM/Gowalla degradation, but no uniform cap reproduces Table II and the
remaining variance is far above the paper's.

An explicit `--dp-noise-stream independent|legacy-global` diagnostic now
isolates the next historical difference. `independent` remains the default
and uses `dp_noise_seed`; `legacy-global` exactly restores revision
`31166ca`'s direct `torch.randn_like` noise generation and is accepted only
under `released-compatibility`. In the latter mode the DP-noise seed is
recorded but not applied. On CUDA, noise consumes the CUDA global RNG while
DataLoader shuffle consumes the CPU global RNG, so these are both historical
global-state behaviors but not literally one shared device stream.

### Seed separation

Random sources are separated into:

- model initialization seed;
- subgraph sampling seed;
- batch sampler seed.
- DP Gaussian-noise seed.
- DP noise stream (`independent` or compatibility-only `legacy-global`).

Checkpoint metadata stores all realized seeds and a SHA-256 hash of sampled
node sets. The DP-noise seed was separated after the first full private matrix
showed correlated seed collapses across datasets and methods. Controlled
non-private diagnosis showed model initialization dominates
Facebook variance; sampling alone was highly stable.

### GRAT variants and diagnostics

- `legacy_released`: released parameterization and current default.
- `paper_appendix_g`: literal Appendix G Equations 37--40 diagnostic.
- `legacy_leaky_attention`: released parameterization with only Equation 40
  edge-score activation changed to LeakyReLU; failed and diagnostic-only.
- `released_default` initialization remains the default.
- `structural_preserving` initialization was tested and failed.
- Training records layer/output/parameter diagnostics, sigmoid saturation,
  microbatch gradient norms, and clipping fractions.

### Input feature protocols

- `ones`: released 32-dimensional all-one input; current default.
- `degree`: exploratory 3-D `[1, normalized indegree, normalized outdegree]`.
  This changes input width and RNG consumption, so it is confounded.
- `degree32`: controlled 32-D
  `[1, normalized indegree, normalized outdegree, 0 x 29]`. It preserves
  parameter shapes, and tests verify identical full initial state dicts for
  paired seeds versus `ones`.

Training, evaluation, summary CSV, checkpoint loading, and resume all honor
the checkpoint feature protocol and input dimension.

## 5. Complete experiment summary

All values below are coverage ratio percentages, mean +/- sample standard
deviation over five runs unless marked as smoke tests.

| Run ID / experiment | Dataset | Result | Status and conclusion |
|---|---|---:|---|
| `validation-non-private` | Email | 46.48 +/- 41.20 | 50-update underfit pilot; reject |
| `validation-non-private` | Facebook | 24.35 +/- 37.63 | 50-update underfit pilot; reject |
| `validation-non-private-v2` | Email | 96.85 +/- 1.58 | 500 updates fixed underfitting |
| `validation-non-private-v2` | Facebook | 88.69 +/- 5.16 | much better, still unstable |
| `validation-non-private-v3` | Email | 96.85 +/- 1.58 | same result under explicit legacy variant |
| `validation-non-private-v3` | Facebook | 88.69 +/- 5.16 | same result under explicit legacy variant |
| `validation-facebook-fixed-sampling` | Facebook | 84.40 +/- 11.52 | fixed sampling; initialization variance visible |
| `diagnostic-facebook-model-only` | Facebook | 84.55 +/- 11.65 | fixed sampling/batch, only model seed changes |
| `diagnostic-facebook-sampling-only` | Facebook | 87.61 +/- 0.08 | fixed model/batch, sampling varies; sampling is not main variance source |
| `validation-facebook-artifact-v2` | Facebook | 84.83 +/- 8.82 | exact author artifact split did not close gap |
| `diagnostic-facebook-released-optimizer` | Facebook | 90.83 +/- 7.10 | lr 0.08, 340 updates, gamma 0.99; closer mean but unstable and not paper default |
| `diagnostic-facebook-structural-init` | Facebook | 32.74 +/- 31.42 | catastrophic; reject |
| `diagnostic-facebook-clipped-non-private` | Facebook | 86.87 +/- 3.30 | zero-noise clipping stabilizes but remains 10.63 below paper |
| `diagnostic-facebook-clipped-t1000` | Facebook | 87.79 +/- 5.90 | only +0.91 mean and worse variance; do not increase T further |
| `diagnostic-facebook-legacy-leaky` | Facebook | 73.80 +/- 21.83 | bimodal saturation collapse; reject |
| `diagnostic-facebook-degree` | Facebook | 87.72 +/- 9.07 | confounded 3-D input experiment; do not use for attribution |
| `diagnostic-facebook-degree32` | Facebook | **91.45 +/- 6.32** | best controlled Facebook feature result; promising but still unstable |
| `diagnostic-email-degree32` | Email | **97.73 +/- 0.93** | validation passed; paper is 98.09 +/- 1.69 |
| `diagnostic-bitcoin-lastfm-degree32` | Bitcoin | **98.81 +/- 0.89** | essentially matches paper 98.99 |
| `diagnostic-bitcoin-lastfm-degree32` | LastFM | **94.44 +/- 2.74** | stable, but 3.49 below paper 97.93 |
| `diagnostic-bitcoin-lastfm-ones-controlled` | Bitcoin | **99.47 +/- 0.72** | beats degree32 and closely matches paper 98.99 |
| `diagnostic-bitcoin-lastfm-ones-controlled` | LastFM | **97.48 +/- 2.13** | beats degree32 and closely matches paper 97.93 |
| `diagnostic-hepph-gowalla-degree32` | HepPh | **79.79 +/- 15.46** | unstable and 18.63 below paper; reject as a uniform feature protocol |
| `diagnostic-hepph-gowalla-degree32` | Gowalla | **87.59 +/- 13.93** | unstable and 12.19 below paper; reject as a uniform feature protocol |
| `diagnostic-hepph-gowalla-ones-controlled` | HepPh | **72.45 +/- 11.06** | unstable and 25.97 below paper; unresolved dataset-specific gap |
| `diagnostic-hepph-gowalla-ones-controlled` | Gowalla | **99.22 +/- 1.33** | closely matches paper 99.78; degree32 is substantially worse |
| `smoke-facebook-degree` | Facebook | 2.88 (1 run, 1 update) | plumbing test only |
| `smoke-facebook-degree32` | Facebook | 20.30 (1 run, 1 update) | plumbing test only |

The validator output `92.77` previously shown for `validation-non-private-v2`
was the average of Email 96.85 and Facebook 88.69, not the Facebook result
alone.

Additional earlier architecture result recorded in the implementation guide:

- literal `paper_appendix_g`: Facebook 93.09 +/- 3.07, but directed Email
  collapsed to 25.65 +/- 19.27. Do not promote it.
- released `StepLR(step_size=5, gamma=0.99)` diagnostic: Facebook
  86.41 +/- 1.12. It reduced variance but lowered mean.
- Xavier initialization diagnostic: Facebook 80.23 +/- 11.33. Reject.

### Critical per-seed Facebook results

| Seed | ones/model-only | clipped T=500 | clipped T=1000 | legacy leaky | degree 3-D | degree32 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 87.55 | 89.99 | 87.55 | 95.82 | 81.47 | 85.65 |
| 1 | 84.45 | 85.71 | 83.91 | 51.66 | 82.32 | 83.91 |
| 2 | 87.36 | 87.77 | 87.36 | 62.37 | 79.63 | 93.38 |
| 3 | 65.73 | 81.72 | 82.52 | 60.41 | 98.32 | 96.93 |
| 4 | 97.66 | 89.17 | 97.59 | 98.73 | 96.83 | 97.40 |

Degree32 preserves identical model initialization and improves Facebook mean
by 6.91 points versus ones, mainly by rescuing seeds 2 and 3. It slightly
reduces seeds 0, 1, and 4, so the benefit is not uniform. It reduces standard
deviation from 11.65 to 6.32, still above the desired threshold of 5.

### Email degree32 per seed

| Seed | ones | degree32 |
|---:|---:|---:|
| 0 | 94.44 | 96.76 |
| 1 | 97.22 | 97.22 |
| 2 | 98.84 | 98.84 |
| 3 | 96.76 | 98.61 |
| 4 | 96.99 | 97.22 |

Degree32 improved Email mean from 96.85 to 97.73 and reduced standard
deviation from 1.58 to 0.93 without meaningful sigmoid saturation.

### HepPh and Gowalla paired feature results

| Seed | HepPh ones | HepPh degree32 | Gowalla ones | Gowalla degree32 |
|---:|---:|---:|---:|---:|
| 0 | 70.99 | 63.58 | 99.82 | 97.34 |
| 1 | 63.70 | 63.58 | 99.81 | 96.37 |
| 2 | 75.42 | 85.41 | 99.81 | 71.58 |
| 3 | 62.38 | 88.74 | 96.84 | 73.22 |
| 4 | 89.78 | 97.65 | 99.82 | 99.48 |

These are controlled pairs: both protocols use 32-dimensional inputs, the
same model seeds, fixed batch seed 0, and identical sampled-node-set hashes
within each dataset. Degree32 improves the HepPh mean by 7.34 points but
increases its standard deviation from 11.06 to 15.46 and remains far below
the paper. On Gowalla, degree32 reduces the mean by 11.62 points and increases
the standard deviation from 1.33 to 13.93. Together with the Bitcoin and
LastFM controls, this completes the six-dataset feature diagnostic and rules
out degree32 as a uniform replacement for the released all-one features.

## 6. Decisions already made

Do not repeat these branches without a new scientific reason:

1. Do not return to 50 optimizer updates; it is severely underfit.
2. Do not blindly increase clipped training beyond 500; T=1000 gave only a
   small mean gain and worse stability.
3. Do not make zero-noise clipping the non-private default. The released
   non-private path uses ordinary SGD, and clipped results remain far below
   Table II.
4. Do not make `structural_preserving` initialization the default.
5. Do not make `legacy_leaky_attention` the default; three seeds collapsed
   with 53%--63% sigmoid saturation.
6. Do not make literal `paper_appendix_g` the default until the missing
   feature/direction convention is resolved; it collapses Email.
7. Do not interpret the 3-D `degree` result as a controlled feature ablation;
   input width changed all later seeded weights.
8. Keep `legacy_released`, `released_default`, ordinary non-private SGD,
   lr 0.005, gamma 1, and T=500 as compatibility defaults for now.
9. Do not promote `degree32` globally. It improved Email/Facebook and the
   unstable HepPh mean, but reduced Bitcoin by 0.65, LastFM by 3.04, and
   Gowalla by 11.62 points versus paired ones controls. Dataset-specific
   feature selection after observing results would not be a uniform paper
   reproduction protocol. The six-dataset diagnostic is complete and `ones`
   remains the uniform reproduction default.

## 7. Latest completed runs and prepared-data state

Latest run directories:

- `outputs/runs/diagnostic-hepph-gowalla-degree32`
- `outputs/runs/diagnostic-hepph-gowalla-ones-controlled`

Both manifests completed all ten model runs and evaluation. The degree32
validator reports aggregate mean 83.69 versus paper mean 99.10, aggregate MAE
15.41, and two unstable configurations. The paired ones validator reports
aggregate mean 85.84, aggregate MAE 13.26, and one unstable configuration.
The aggregate again hides a dataset split: all-one features reproduce Gowalla
well at 99.22 +/- 1.33, while HepPh remains severely low and unstable at
72.45 +/- 11.06. The six-dataset controlled feature comparison is now
complete; `ones` remains the global compatibility default, and the HepPh gap
requires diagnosis independent of post-hoc feature selection.

All six prepared datasets are on the current explicit protocol:
`active`, self-loops removed, generated split, seed 0:

| Dataset | Current prepared state | Train nodes | Test nodes |
|---|---|---:|---:|
| Email | active, self-loops removed | 462 | 458 |
| Bitcoin | active, self-loops removed | 2,033 | 2,378 |
| LastFM | active, self-loops removed | 3,096 | 3,196 |
| Facebook | active, self-loops removed | 10,011 | 10,080 |
| HepPh | active, self-loops removed | 5,249 | 5,264 |
| Gowalla | active, self-loops removed | 79,978 | 80,392 |

Revalidate the latest paired runs with:

```bash
.venv/bin/python experiments/validate_results.py \
  outputs/runs/diagnostic-hepph-gowalla-degree32/evaluation/summary.csv
.venv/bin/python experiments/validate_results.py \
  outputs/runs/diagnostic-hepph-gowalla-ones-controlled/evaluation/summary.csv
```

## 8. Work that has not been completed

- The complete private matrix (epsilon 1--6, PrivIM/PrivIM+SCS/PrivIM*) has
  not been run under the final code and seed protocol.
- The non-private representation protocol is settled: retain `ones` uniformly
  for the private matrix. HepPh remains a known non-private reproduction gap
  and must be called out when interpreting its private results.
- External competitors EGN, HP, and HP-GRAT are not fully reproduced locally.

## 9. Remaining scientific ambiguities

1. **Node feature matrix X:** the paper defines X but does not publish its
   experimental construction. Released code uses all-one features. Degree32
   is an explicit inference/diagnostic, not a paper fact.
2. **Training iterations T:** named by Algorithm 2 but not numerically
   published. T=500 is an empirical repository choice.
3. **Non-private clipping:** Section V-A calls Non-Private PrivIM* with
   epsilon infinity. A literal Algorithm 2 limit retains clipping and removes
   noise, while released code bypasses clipping. Both are explicit; ordinary
   SGD remains default.
4. **GRAT equations versus released layer:** Appendix G transforms features
   before attention and uses LeakyReLU; released code attends over untransformed
   inputs, uses ReLU, and transforms after aggregation with bias.
5. **Privacy accountant sampling law:** Theorem 3 prints a binomial mixture,
   while Algorithm 2 describes uniform fixed-size batches. Both named modes
   remain explicit.

## 10. Migration checklist

Before leaving the current device:

1. Push local `main`. A clone made before pushing will stop at the old remote
   commit and omit the reproduction implementation and migration records.
2. The committed repository includes source/docs/tests, deterministic prepared
   splits, raw-data checksums, and curated manifests/summaries. Additionally
   copy when needed:
   - `PrivIM_Report.pdf`;
   - `AGENTS.md`, `README.md`, `docs/`;
   - `dataset/`, `experiments/`, `model/`, `tests/`;
   - `requirements.txt`;
   - the complete `outputs/runs/` when checkpoints and per-seed diagnostics
     must be retained (only curated records are committed);
   - either the 31 GB raw data or a plan to redownload it.
3. Verify raw files with `sha256sum -c raw_dataset/checksums.sha256` before
   regenerating prepared data.
4. Do not copy `.venv`; recreate it from requirements with CUDA/PyTorch/DGL
   builds appropriate for the new machine.
5. On the new machine run:

```bash
sha256sum PrivIM_Report.pdf
.venv/bin/python -m unittest discover -s tests
git status --short
```

6. Verify the paper hash equals the value at the top of this document and all
   31 tests pass before resuming experiments.
7. Read this document and `docs/paper_implementation_guide.md` before making
   further algorithmic changes.

## 11. Standard commands

Inspect a run:

```bash
.venv/bin/python experiments/validate_results.py \
  outputs/runs/<run-id>/evaluation/summary.csv
```

Current ordinary non-private controlled baseline pattern:

```bash
.venv/bin/python experiments/run_reproduction.py \
  --dataset <datasets...> \
  --only-non-private \
  --round 5 \
  --device auto \
  --skip-prepare \
  --grat-variant legacy_released \
  --grat-initialization released_default \
  --feature-protocol ones \
  --model-seed 0 --model-seed-mode per-round \
  --sampling-seed 0 --sampling-seed-mode fixed \
  --batch-seed 0 --batch-seed-mode fixed \
  --run-id <unique-run-id>
```

Replace `ones` with `degree32` only for the explicit feature diagnostic.

The eventual full paper matrix command remains:

```bash
.venv/bin/python experiments/run_reproduction.py \
  --device auto \
  --run-id <unique-full-run-id>
```

Use `--dry-run` first. This full command is intentionally deferred until the
feature protocol is settled.

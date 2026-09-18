# PrivIM experiment and implementation handoff

Last updated: 2026-09-18 (Asia/Shanghai)

This document is the durable handoff for continuing the PrivIM reproduction
on another machine. Read it together with `AGENTS.md`,
`PrivIM_Report.pdf`, and `docs/paper_implementation_guide.md`. The PDF remains
the primary technical specification.

## 1. Critical repository state

- Branch: `main`
- Base commit: `f4a5ea303e318ca45ca233728a4f48e38f6e2f21`
- Base commit subject: `数据处理模块`
- Paper SHA-256:
  `cf7fc97430366bec111d230d1a945a79c844980dd791373c1d04374514c647ad`
- Current test status: **31 tests passing** with
  `.venv/bin/python -m unittest discover -s tests`.
- The working tree contains many modified and untracked files. The work in
  this handoff is **not represented by the base commit alone**. Copy the
  complete working tree or commit it before moving machines.
- `outputs/` is about 27 MB, `dataset/prepared/` about 15 MB,
  `raw_dataset/` about 31 GB, and `.venv/` about 6.4 GB on the current machine.
  Recreate `.venv` on the new device rather than copying it.
- Experimental outputs and raw data may be ignored by Git. Committing source
  changes does not replace separately copying `outputs/runs/`, prepared data,
  and any raw datasets that should be retained.

Current modified tracked files include `README.md`, dataset preparation,
training/evaluation, GRAT, loss, and requirements files. Important untracked
files include `AGENTS.md`, the complete `docs/` directory,
`experiments/config.py`, `experiments/privacy.py`,
`experiments/run_reproduction.py`, `experiments/sampling.py`,
`experiments/seeding.py`, `experiments/validate_results.py`,
`model/ClassicalGNN.py`, `tests/`, and `raw_dataset/`.

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

### Seed separation

Random sources are separated into:

- model initialization seed;
- subgraph sampling seed;
- batch sampler seed.

Checkpoint metadata stores all realized seeds and a SHA-256 hash of sampled
node sets. Controlled diagnosis showed model initialization dominates
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
9. Treat `degree32` as the leading candidate feature diagnostic, not yet a
   paper-claimed or global default.

## 7. Current interrupted run

Run directory:
`outputs/runs/diagnostic-bitcoin-lastfm-degree32`

State at handoff:

- data preparation: completed;
- Bitcoin training: manifest says `running`, but no process is currently
  running;
- Bitcoin seed 0 checkpoint completed (500 updates, degree32);
- Bitcoin seeds 1--4: not completed;
- LastFM training: pending;
- evaluation: pending.

Bitcoin and LastFM prepared data are now on the current explicit protocol:
`active`, self-loops removed, generated split, seed 0. Resume with:

| Dataset | Current prepared state | Train nodes | Test nodes |
|---|---|---:|---:|
| Email | active, self-loops removed | 462 | 458 |
| Bitcoin | active, self-loops removed | 2,033 | 2,378 |
| LastFM | active, self-loops removed | 3,096 | 3,196 |
| Facebook | active, self-loops removed | 10,011 | 10,080 |
| HepPh | legacy metadata; must reprepare | unknown in legacy metadata | unknown |
| Gowalla | legacy metadata; must reprepare | unknown in legacy metadata | unknown |

```bash
.venv/bin/python experiments/run_reproduction.py \
  --dataset bitcoin lastfm \
  --only-non-private \
  --round 5 \
  --device auto \
  --grat-variant legacy_released \
  --grat-initialization released_default \
  --feature-protocol degree32 \
  --model-seed 0 \
  --model-seed-mode per-round \
  --sampling-seed 0 \
  --sampling-seed-mode fixed \
  --batch-seed 0 \
  --batch-seed-mode fixed \
  --run-id diagnostic-bitcoin-lastfm-degree32 \
  --resume
```

The current launcher resumes at command granularity, not individual round
granularity. Because only 1 of 5 Bitcoin checkpoints exists, it will relaunch
the five-round Bitcoin training command and overwrite seed 0 before continuing.

Validate afterward:

```bash
.venv/bin/python experiments/validate_results.py \
  outputs/runs/diagnostic-bitcoin-lastfm-degree32/evaluation/summary.csv
```

Acceptance target: mean within 3 points of Table II and standard deviation
below 5. If Bitcoin and LastFM pass, prepare and test HepPh/Gowalla next.
Their current prepared metadata is legacy and will fail `--skip-prepare`, so
the first new run must omit `--skip-prepare`.

## 8. Work that has not been completed

- Degree32 cross-dataset validation is incomplete for Bitcoin, LastFM, HepPh,
  and Gowalla.
- No decision has been made to promote degree32 to a default.
- The complete private matrix (epsilon 1--6, PrivIM/PrivIM+SCS/PrivIM*) has
  not been run under the final code and seed protocol.
- Full DP results should not be launched until the non-private representation
  protocol is settled.
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

1. Commit all source/docs/tests changes, or archive the complete working tree.
   A plain checkout of base commit `f4a5ea3` will lose nearly all work.
2. Copy at minimum:
   - `PrivIM_Report.pdf`;
   - `AGENTS.md`, `README.md`, `docs/`;
   - `dataset/`, `experiments/`, `model/`, `tests/`;
   - `requirements.txt`;
   - `outputs/runs/` including manifests, JSON, CSV, and checkpoints;
   - `raw_dataset/README.md` and either the 31 GB raw data or a plan to
     redownload it.
3. Copy `dataset/prepared/` if exact current splits are required. Metadata and
   source hashes will allow preflight verification.
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

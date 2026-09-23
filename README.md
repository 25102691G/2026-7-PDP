# PrivIM reproduction

For the complete current implementation state, experiment history, rejected
branches, partial runs, and machine-migration checklist, read
[`docs/experiment_handoff.md`](docs/experiment_handoff.md) first.

This repository reproduces the main method and ablations from
“PrivIM: Differentially Private Graph Neural Networks for Influence
Maximization”. The paper is included as `PrivIM_Report.pdf`.

## Environment

The tested environment is Python 3.8, PyTorch 2.1.1, DGL 2.0.0 and igraph
0.11.5. For CPU execution:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For CUDA execution, install PyTorch and DGL builds matching the local CUDA
runtime first, then install the remaining requirements. The training and
evaluation commands accept `--device auto`, `cpu`, or `cuda`.

The privacy optimizer and two explicit RDP interpretations are implemented in
`experiments/privacy.py`: the default `paper_binomial` follows the mixture
printed in Theorem 3, while `fixed_batch_hypergeometric` follows Algorithm
2's uniformly sampled fixed-size batch as an explicit implementation variant.
The selected mode, calibrated sigma,
achieved epsilon, sensitivity and effective noise scale are stored with every
checkpoint.

## Dataset preparation

Place the six raw datasets in the layout documented by
`raw_dataset/README.md`, then run:

```bash
.venv/bin/python dataset/prepare_datasets.py all \
  --seed 0 --train-ratio 0.5 --isolated-policy active \
  --remove-self-loops --force
```

For each dataset this creates:

- train/test induced edge lists with contiguous local IDs;
- `train_nodes.csv` and `test_nodes.csv` under the selected isolated-node policy;
- `node_mapping.csv`;
- `metadata.json` with direction, split settings, statistics, and source hash.

The reproduction launcher defaults to `active`, matching an edge-list
pipeline in which nodes that become isolated after the 50/50 split are not
model candidates. Use `--isolated-policy preserve` for the alternative
interpretation. The policy is recorded and validated before training.
Training and evaluation preserve the paper's directed/undirected distinction.
Friendster is not
handled by this in-memory preparation command because the paper partitions it
with an external-memory process.

## Training

The three paper configurations are:

```bash
# Algorithm 1: theta-bounded PrivIM
.venv/bin/python experiments/train_models.py --dataset bitcoin --settings PrivIM --target_epsilon 3

# Algorithm 3, Stage 1 only
.venv/bin/python experiments/train_models.py --dataset bitcoin --settings PrivIM+SCS --target_epsilon 3

# Algorithm 3, SCS + BES (PrivIM*)
.venv/bin/python experiments/train_models.py --dataset bitcoin --settings PrivIM+ --target_epsilon 3
```

Defaults follow the paper: `q=256/|V_train|`, `L=200`, `tau=0.3`, `theta=10`,
three GNN layers with 32 hidden units, learning rate 0.005, and five runs.
GRAT defaults to the released layer (`--grat-variant legacy_released`) used by
the Table II implementation. The literal Appendix G Equations 37-40 variant
is available as `paper_appendix_g` for diagnosis, but it is not the default:
with the released all-one node inputs it improved Facebook while collapsing
the directed Email baseline. Historical checkpoints are detected and loaded
automatically.
Use `--grat-variant legacy_leaky_attention` to isolate Equation 40's
LeakyReLU edge-score activation while retaining all other released-layer
parameterization and aggregation behavior. It is diagnostic-only: a
controlled Facebook five-seed run produced `73.80 +/- 21.83`, substantially
worse and less stable than `legacy_released`.
Node inputs default to the released 32-D all-one convention
(`--feature-protocol ones`). Because the paper defines `X` without publishing
its experimental construction, structural features are available only as
diagnostics. `degree` is the original 3-D exploration; `degree32` pads
`[1, normalized in-degree, normalized out-degree]` with 29 zeros so the model
shape and seeded initialization remain identical to `ones`. Checkpoint
metadata makes the choice explicit and evaluation reconstructs the matching
feature representation.
Use `--auto-parameters` to select `n` and `M` from the paper's grids using its
Gamma indicator and the original graph size `|V|` recorded in `metadata.json`.
Use `--no-dp` for the non-private variant. The effective theta option is
`--theta` (`--K` remains a backwards-compatible alias).
The paper calls this baseline PrivIM* with `epsilon = infinity`, but does not
say whether clipping is removed along with noise. Add `--clip-non-private` to
test the literal zero-noise limit of Algorithm 2: each subgraph gradient is
still clipped to `--clip-norm`, while privacy accounting and Gaussian noise
remain disabled. This mode is intentionally not the compatibility default;
per-subgraph backward passes are substantially slower, and it must pass the
five-run baseline validation before becoming the reproduction default.

Direct training commands write below `outputs/`. The full launcher writes each
experiment below `outputs/runs/<run_id>/`, with a manifest that fixes its data,
privacy and command provenance.

Sampling is independently named. `--sampling-protocol paper` is the default
and implements Algorithms 1 and 3. For provenance diagnosis only,
`--sampling-protocol released-legacy` reproduces revision `31166ca`'s retry
sampler, bidirectional traversal of directed graphs, and capped boundary
stage. It is accepted only together with
`--protocol-profile released-compatibility`; its checkpoints explicitly
refuse a strict privacy claim. Keeping these switches independent prevents a
released-code diagnostic from silently replacing the paper algorithm.
For the BES/update-count isolation experiment, the launcher additionally
accepts `--compatibility-update-cap N`. It is valid only under
`released-compatibility`, truncates the historical full-epoch traversal after
`N` optimizer updates, and deliberately leaves its 20-step noise calibration
unchanged. The cap is diagnostic metadata, not a privacy repair.

Random sources are explicit and separated. By default, all five rounds reuse
one subgraph container (`--sampling-seed 0 --sampling-seed-mode fixed`), while
model initialization and batch selection use independent per-round sequences
from `--model-seed 0 --model-seed-mode per-round` and
`--batch-seed 0 --batch-seed-mode per-round`. Private runs additionally use
an independent Gaussian-noise sequence from
`--dp-noise-seed 0 --dp-noise-seed-mode per-round`. Use
`--sampling-seed-mode per-round` only when measuring end-to-end sampling
variability. To isolate sampling alone, set sampling to `per-round` and both
model/batch modes to `fixed`; also hold DP noise fixed for private runs.
Checkpoint metadata records all four realized seeds and a hash of the sampled
node sets. Runs created before this protocol
require a new run ID; they are not resumed in place.

For released-code provenance only, `--dp-noise-stream legacy-global`
restores revision `31166ca`'s direct `torch.randn_like` calls. It is accepted
only with `--protocol-profile released-compatibility`; the default
`independent` stream remains the reproducible protocol. In legacy-global mode
the separately supplied DP-noise seed is recorded but not applied because the
noise consumes PyTorch's already-seeded global RNG for the tensor device.

On multi-GPU hosts, pin each reproduction process to a physical GPU with
`--gpu-id N`. The launcher sets `CUDA_VISIBLE_DEVICES=N` for all of that run's
training and evaluation subprocesses, where the selected card is exposed as
logical `cuda:0`. Use a distinct `--run-id` for every concurrent process.

## Evaluation

The paper reports one-step IC influence with edge probability 1 and `k=50`:

```bash
.venv/bin/python experiments/evaluate_models.py --dataset bitcoin --settings PrivIM+ --k 50
```

Per-run CSV files retain their full experiment path. The command also writes
`outputs/evaluation/summary.csv` containing the mean and sample standard
deviation over runs. Seed selection uses pre-sigmoid logits, which is
mathematically order-equivalent to probabilities but avoids numerical ties
introduced by sigmoid saturation. Equal logits use ascending node ID as a
deterministic secondary key. Evaluation records the saturation fraction,
top-k boundary tie count, logit dispersion, and top-k margin; a large tie count
still indicates genuine model collapse rather than an evaluation fix.

## Full experiment launcher

The following runs dataset preparation, all six privacy budgets, five seeds,
and the PrivIM/SCS/BES ablations for all six datasets:

```bash
.venv/bin/python experiments/run_reproduction.py \
  --device auto \
  --run-id paper-v1
```

This is a large experiment. Inspect the exact commands with `--dry-run`; use
`--dataset`, `--epsilon`, and `--settings` to select a smaller matrix. Resume a
partially completed isolated run with `--run-id paper-v1 --resume`. Evaluation
reads only the checkpoints named by that run root, so unrelated historical
models are not mixed into the summary.

The reproduction default is 500 optimizer updates per run (`--iterations
500`). Algorithm 2 calls this quantity `T`, but the paper does not publish a
numerical value. A 50-update pilot was severely underfit: on Email, increasing
only `T` from 50 to 500 changed the five-run non-private result from
`46.48 +/- 41.20` to `96.85 +/- 1.58`. The selected value is therefore an
explicit, empirically validated reproduction parameter rather than a value
claimed by the paper. It is stored in every checkpoint and run manifest.
The paper reproduction keeps the reported learning rate fixed at `0.005`
(`--lr-decay 1`). The released implementation's
`StepLR(step_size=5, gamma=0.99)` is available as a compatibility experiment;
it reduced Facebook variance in a controlled run but also reduced mean
coverage. The selected schedule is stored with every checkpoint.

For strict-DP stability diagnostics, the launcher exposes batch size and
optional exponential moving averaging of private optimizer iterates:

```bash
.venv/bin/python experiments/run_reproduction.py \
  --dataset email --epsilon 4 --settings PrivIM+ \
  --iterations 200 --batch-size 8 \
  --parameter-averaging ema --ema-decay 0.99 \
  --skip-prepare --skip-non-private \
  --run-id diagnostic-email-t200-b8-ema
```

EMA is DP post-processing and does not change the accounted epsilon, but it is
an explicit diagnostic rather than the paper default. It is rejected by the
released-code compatibility profile. Use separate run IDs when comparing
`T` and batch-size combinations.

The launcher also exposes `--optimizer-momentum` as a strict-DP optimizer
diagnostic. Momentum is applied only after private-gradient construction, so
it does not change privacy accounting, but it is not a reported paper
default. A controlled `epsilon=4`, `T=200`, momentum `0.9` experiment reduced
utility and increased variance on Facebook and Gowalla; the strict default
therefore remains `0`.

The paper's Equation (5) leaves the penalty coefficient `lambda` unspecified.
The released code uses `lambda=1`, which remains the uniform default. The
launcher exposes it as `--loss-penalty` and stores it as `loss_lambda` in each
checkpoint. A controlled HepPh diagnostic found that `lambda=0.5` improves
Non-Private from `73.76 +/- 9.20` to `92.12 +/- 7.98` and strict epsilon 4
PrivIM* from `61.63 +/- 23.63` to `90.68 +/- 6.15`. At epsilon 1 it reaches
only `48.04 +/- 31.45`, so this is a dataset-specific diagnostic, not a new
global reproduction default.

### Released-code compatibility diagnostic

To reproduce the audited behavior of released revision `31166ca` without
changing the strict defaults, use the isolated profile:

```bash
.venv/bin/python experiments/run_reproduction.py \
  --protocol-profile released-compatibility \
  --dataset email facebook gowalla \
  --epsilon 4 \
  --settings PrivIM PrivIM+SCS PrivIM+ \
  --device auto \
  --skip-prepare --skip-non-private \
  --run-id released-compatibility-eps4
```

This profile retains the current tested Algorithms 1/3 samplers and prepared
data, but intentionally restores the released training/privacy behavior: 20
full DataLoader epochs while privacy
noise is calibrated for 20 steps, base `Ng=10`, private-SGD momentum
`0.9`, `StepLR(gamma=0.99, step_size=5)`, the fixed-batch accountant, the
released sensitivity/noise factor of 2, fixed model/batch/noise streams, and
one Python sampling stream advancing across rounds. It is a provenance
diagnostic, not a strict paper epsilon run. Checkpoint metadata records
`privacy_accounting_steps`, `actual_optimizer_steps`,
`actual_steps_epsilon_nominal`, and
`actual_steps_epsilon_effective_noise`; `strict_privacy_claim` is always false
for this profile. A one-round Email PrivIM* smoke run completed 40 actual
updates from 20 configured epochs/20 calibration steps and reached 98.84%
coverage, demonstrating
why this distinction materially affects Table II comparisons.
The released CLI declared `r=1`, but applying that value literally to paper
Algorithm 1 with `theta=10` cannot produce its `n=40` subgraphs (the bounded
one-hop neighborhood has at most 11 vertices). The compatibility profile
therefore retains the tested paper sampler at `r=3` and records the released
manual base bound `Ng=10` as a separate, intentionally inconsistent override.

`--skip-prepare` now performs a strict preflight. Legacy prepared data that do
not record the requested isolated-node and self-loop policies are rejected;
rerun preparation once rather than silently mixing protocols.

For staged validation, first run one non-private method, then epsilon 4:

```bash
.venv/bin/python experiments/run_reproduction.py \
  --dataset email facebook --only-non-private \
  --round 5 --run-id validation-non-private

.venv/bin/python experiments/run_reproduction.py \
  --dataset email facebook --settings PrivIM PrivIM+SCS PrivIM+ \
  --epsilon 4 --round 5 --run-id validation-eps4
```

Audit a completed summary against Table II and flag unstable configurations:

```bash
.venv/bin/python experiments/validate_results.py \
  outputs/runs/validation-eps4/evaluation/summary.csv
```

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

The tests cover theta projection, fixed-size/r-hop RWR, the frequency bound,
BES accounting, both isolated-node policies, DP-SGD clipping, named privacy
accountants, explicit sensitivity selection, and accountant monotonicity.
They also cover the released compatibility profile, epoch-to-update expansion,
continuous sampling stream, and resume isolation between protocol profiles.

## Implemented model scope

The main experiments use GRAT. The appendix architecture study can use
`GRAT3`, `GCN3`, `GAT3`, `GIN3`, or `GraphSAGE3` through `--model_name`.
CELF is computed during evaluation. EGN and HP are external competitor
implementations and are not relabeled as PrivIM implementations in this repo.

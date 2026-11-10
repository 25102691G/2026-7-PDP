
# PrivIM / PDP-PrivIM Research Project

## Research Objective
Study whether node-level Personalized DP can improve influence-maximization
Coverage over a uniform strictest Node-DP policy while meeting heterogeneous
node requirements, and measure the utility gap to a relaxed uniform policy
and a Non-Private reference. This is a research hypothesis, not a proven
utility ordering; distinguish mathematical guarantees, observed development
results, and unfinished independent confirmation.

## Core Constraints
- Preserve the existing PrivIM reproduction.
- Do not modify GRAT, the original loss function,
  IC diffusion, SCS, BES, or DP-SGD unless the
  current task explicitly authorizes it.
- Prefer minimal changes over refactoring.
- Keep original DP and new PDP implementations
  independently runnable.
- Never silently change experimental parameters.
- Never claim a mathematical privacy guarantee
  based only on passing code tests.
- Never change the privacy accountant without
  explicitly documenting and reviewing the
  corresponding theoretical assumptions.

## Workflow
- Inspect relevant code before editing.
- Explain the expected scope of changes.
- Implement one small task at a time.
- Run focused tests after each modification.
- Check backward compatibility with PrivIM.
- Report changed files, tests, and limitations.
- Do not commit or push without approval.

## Dataset
- Preserve the five thesis datasets: Email, Bitcoin, LastFM, HepPh and Facebook; use the existing `dataset/prepared/` files.
- Keep the original 50:50 training/test split, node ID mappings and graph structures unchanged.
- Do not download, repartition or preprocess the datasets.
- Strictly isolate training and testing graphs; do not run Gowalla.

## Research Experiments
- Follow the original PrivIM paper-style workflow for models, baselines, ablations and repeated experiments.
- Save each experiment under an independent run ID; record public model/partition seeds and complete hyperparameters.
- Do not record secret Bernoulli/Gaussian seeds, random bytes, retained masks or noise vectors.
- Report n=1 with std=NA; repeated experiments must report actual n, mean and sample standard deviation.
- Preserve all outputs and low-utility seeds; no Coverage-based seed selection, deletion or retry-until-success.
- Development initialization seeds may be 0..4; no extra sampling-seed search.

## Privacy
- Preserve Node-DP/PDP mathematical proofs and privacy accounting.
- Scientific experiments and formal private-model publication are different uses.
- Research may train multiple models; distinguish single-model guarantees from joint-publication composition guarantees.
- Do not claim strict DP/PDP certification for an uncertified machine implementation.
- Never reset production Ledgers, forge authorization or change epsilon to claim cumulative privacy targets still hold.

## U1 Forward Main Experiment Decision (2026-11-04)
- This authorized decision supersedes earlier prospective main-table membership only. All existing mechanism, data, evaluation, privacy and evidence-preservation constraints remain. Older protocols/reports below are historical archives and must not be rewritten or used as a second current main-table definition.
- Three method families, four configurations: Non-Private; Uniform Node-DP epsilon=1; PDP v2 with heterogeneous targets1/2/4; Uniform Node-DP epsilon=4.
- Non-Private: unchanged original GRAT3/frozen training and evaluation; no DP/PDP protection, utility reference only.
- Uniform Node-DP epsilon=1: existing full2340-coordinate training, sigma_sum486, p_v=1, T100, same architecture/doubleClip3/K2/B16/lr.005 and post-update32-weight projection as v2; bias/initialization not projected. Conditional ideal bound epsilon<=0.998694<1, delta=1e-5. This is the primary fair strict-policy control; shared policy caps do not mean equal actual per-node losses.
- PDP v2 remains the main algorithm, unchanged: full2340 private coordinates, sigma_sum260, frozen heterogeneous Bernoulli/Registry, doubleClip3/K2/T100 and post-update32-weight projection. Proven conditional ideal upper bounds are approximately(0.986940353797,2.6894142137e-6) for target1 and(1.982092738992,1e-5) for target2/4. Budget4 does not use all additional allowance beyond2.
- Uniform Node-DP epsilon=4: RELAXED_PRIVACY_REFERENCE, prospective target epsilon<=4/delta<=1e-5 with p_v=1 and the same full2340 training/projection rules. It does not establish compliance with the original target1/2 policies; it is neither a fair same-policy baseline nor a theoretical Coverage upper bound. No utility ordering is assumed. A separate full100-step/Delta12 Gaussian calibration, implementation/proof review and explicit authorization are required; sigma260/PDP budget4 labels or PrivIM epsilon4 must not be relabelled as this configuration.
- Evidence status: Non-Private25/25, Uniform epsilon1 25/25 and PDP v2 25/25 existing development runs verified; Uniform epsilon4 0/25 qualifying runs, NOT_AVAILABLE. THREE_CONFIGURATIONS_COMPLETE; UNIFORM_EPS4_EXPERIMENT_MISSING; NEW_DESIGN_PARTIALLY_SATISFIED. Historical100-row matrices do not make this new matrix100/100 complete. See docs/pdp_research/stage_u1_experiment_design_audit/REPORT.md.
- Future main tables exclude New Matched PDP (HISTORICAL_PRIVACY_MATCHED_FROZEN_BACKBONE_REFERENCE), PDP v1 (HISTORICAL_NO_PROJECTION_ABLATION), Uniform-v2 (HISTORICAL_EQUIVALENT_UNIFORM_MECHANISM_VERIFICATION), and G3/C3 (HISTORICAL_REJECTED_DEVELOPMENT_CANDIDATES). Preserve all their code, checkpoints, configurations, results, low-utility runs and failures; never hide New Matched results because they exceed v2 on some datasets.
- Report every seed, mean Coverage, sample SD, minimum/maximum, strict<50% frequency, attempts/failures/retries/unexecuted tasks. Beating New Matched PDP or matching a fixed PrivIM-paper SD is not an automatic acceptance requirement. Do not assert PDP necessarily outperforms either Uniform configuration.
- This decision authorizes audit/governance changes only: no new real training/inference, epsilon4 training-code implementation, independent final confirmation or paper writing; no commit/push. Missing experiments require later explicit approval.

## Historical Experimental Matrix (through 2026-11-03; superseded above for future main tables)
- At that stage, Non-Private, Uniform Node-DP, Matched PDP and PDP v2 were the formal methods.
- PrivIM* is historical only; preserve reproduction code, PrivIM复现.md and all records.
- Non-Private: unconstrained performance reference; no DP/PDP guarantee or equal-privacy comparison.
- Uniform Node-DP: G0 stable slots; uniform budget meets the strictest policy; report actual epsilon/delta upper bounds. Same policy caps do not imply identical per-node proven losses.
- New Matched PDP: matched_pdp_frozen_backbone_v1; public-seeded frozen GRAT3 backbone, 2307 frozen coordinates and 33 privately trained final fc coordinates; post-update projection of 32 weights only. Five-dataset seeds 0..4 development (25 one-shot fits) completed and integrity-verified; see docs/pdp_research/new_matched_pdp_development/REPORT.md. Original PDP v1 remains HISTORICAL_NO_PROJECTION_ABLATION with all artifacts preserved.
- PDP v2: frozen heterogeneous budgets, stable slots, double clipping and final 32-weight nonnegative projection; retain archived conditional ideal PDP proof.
- At that stage, the four-method development matrix was updated; stability and independent confirmation remain unaccepted, and paper writing remains blocked. No further training is authorized by this status.
- Detailed parameters, reuse decisions and acceptance gates: docs/pdp_research/FIVE_DATASET_EXPERIMENT_PROTOCOL.md.

## Theory and Stage Gates
- Aim for mathematical Node-DP/PDP guarantees, not machine-level information-theoretic certification.
- Prove or cite node add/remove adjacency, fixed identities/Registry, sensitivity, Gaussian/RDP accounting, adaptive composition, personalized Bernoulli amplification, per-node epsilon/delta bounds and training/release boundaries.
- Conditional fixed-prepared-domain proofs do not cover raw preprocessing; joint model publication requires composition beyond single-model budgets.
- Seek higher mean Coverage AND lower sample SD, severe-low-utility frequency and extreme random-trajectory losses; retain every low-utility run.
- G0: conditional ideal PDP proof archived. G1/G2: substantial random-trajectory instability observed. G3: 100-step parameter averaging failed the five-dataset screen; not formal PDP v2.
- Freeze mechanism and acceptance criteria before independent confirmation: recommend >=10 new predetermined initialization seeds per dataset/configuration (20 if resources permit); development seeds 0..4 are not independent confirmation.
- Each stage requires acceptance before the next; no paper writing until necessary stability research, fair four-configuration comparisons and independent final confirmation finish.

## Execution Discipline
- No new large-scale real training or historical deletion without approval.
- Do not alter frozen algorithms to improve Coverage, production privacy ledgers or consumption records.
- No commit/push and no premature paper draft; finish authorized preparation and wait for the next instruction.

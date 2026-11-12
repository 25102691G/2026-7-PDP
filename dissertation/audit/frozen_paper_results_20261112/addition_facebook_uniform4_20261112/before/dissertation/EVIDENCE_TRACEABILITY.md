## Current table authority — 2026-11-12

Table6.1 now contains14researcher-fixed selected development configurations, n=3 each,42fits, from the480-fit screen. Other cells are omitted, not unexecuted. Frozen source mapping: audit/frozen_paper_results_20261112/FROZEN_RESULTS.json; generator: scripts/verify_frozen_paper.py; parameters and seed triplets: tables/frozen_parameters.tex and tables/frozen_seed_rows.tex. This supersedes earlier main-table definitions below; all historical evidence remains preserved. No further training or result/configuration changes for those14dataset/method cells.

# Current five-seed draft evidence map

This revision supersedes the previous manuscript data scope on direct researcher instruction. No historical experimental protocol or record has been rewritten.

| Claim | Source | Limit |
|---|---|---|
| n5, seeds0..4, all100 raw outcomes and 20 cell summaries | U5 PER_SEED_RECHECK.csv and its original final evaluations; scripts/verify_five_seed.py | Existing development provenance retained, no independent confirmation claim |
| Fixed model/data/loss/evaluator | Original source, U4 SOURCE_INDEX.md, existing configuration records | No graphs or checkpoints loaded |
| Signal norm bound and noise scales | experiments/node_dp_slots/aggregation.py, mechanism.py; projected policy; uniform specs; U4 GAUSSIAN_NOISE_ANALYSIS.md | Parameter-space scale, not measured functional SNR or Coverage variance |
| Conditional Gaussian path effects | G2 controlled report and witnesses; U4 G2_INDEPENDENT_SCALAR_RECHECK.json | Historical limited conditions, not population variance decomposition |
| Ranking counterexamples | U4 OPTIMIZATION_RANKING_REVIEW.md, RANKING_SCALAR_REVIEW.csv; G1 historical report | Boundary ties do not explain every severe loss |
| Conditional mathematical privacy | Original G0/U0/U2 proofs and certificates | Unchanged; no machine or joint-release certification |
| Current manuscript tables/plot | tables/coverage.tex, five_seed_rows.tex; figures/five_seed_coverage.pdf | All five observations, SD not SE/CI |

See INSTABILITY_DIAGNOSIS.md for checked observations and unresolved questions. Historical audits and the prior manuscript snapshot are internal history, not numerical authority for this revision. No related large-scale continuation is planned.

Current manuscript reporting scope (2026-11-12): five datasets x four configurations x three runs per cell = 60 runs. Table 6.1 reports Coverage mean and sample SD. Current individual outcomes, actual public seed IDs and parameters are in the Appendix B tables; the development/source protocol and full historical cohorts are retained there separately. Frozen numerical outcomes and original provenance are unchanged. Earlier revision descriptions below are historical unless restated here.

Current Table6.1 authority (2026-11-12):20cells/60displayedfits;15complete seed0/1/2 configurations plus5explicit post hoc three-run subsets from U6 historical cohorts. Source sample sizes40/200/200/200/140 and full mean/SD/min/max/<50 summaries are shown in tables/subset_source_cohorts.tex;780raw outcomes are frozen in audit/frozen_paper_results_20261112/U6_FULL_SOURCE_COHORTS.json. Actual public seeds are in tables/frozen_seed_ids.tex. verify_frozen_paper.py verifies all60selected values and all780source-cohort values. Supersedes older current-table descriptions below; no new experiment.

## Current table authority — 2026-11-12

Table6.1 now contains20researcher-selected cells, n=3 each,60fits, from the480-fit screen and144-fit follow-up. Other cells are omitted, not unexecuted. Frozen source mapping: audit/frozen_paper_results_20261112/FROZEN_RESULTS.json; generator: scripts/verify_frozen_paper.py; parameters and seed triplets: tables/frozen_parameters.tex and tables/frozen_seed_rows.tex. This supersedes earlier main-table definitions below; all historical evidence remains preserved. No further training or result/configuration changes for those20dataset/method cells.

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

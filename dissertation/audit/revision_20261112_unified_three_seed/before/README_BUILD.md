Current Table6.1 authority (2026-11-12):20cells/60displayedfits;15complete seed0/1/2 configurations plus5explicit post hoc three-run subsets from U6 historical cohorts. Source sample sizes40/200/200/200/140 and full mean/SD/min/max/<50 summaries are shown in tables/subset_source_cohorts.tex;780raw outcomes are frozen in audit/frozen_paper_results_20261112/U6_FULL_SOURCE_COHORTS.json. Actual public seeds are in tables/frozen_seed_ids.tex. verify_frozen_paper.py verifies all60selected values and all780source-cohort values. Supersedes older current-table descriptions below; no new experiment.

# PDP-PrivIM dissertation: fixed selected paper results

Current scope (2026-11-12): Table6.1 retains20researcher-selected cells,60fits displayed;15complete seed0/1/2 configurations and5post hoc subsets. They are selected development configurations from480initial and144follow-up fits, not independent confirmation. All20cells are filled and all original evidence remains preserved. The20selected dataset/method cells must not be trained again or have their selected parameters/results changed. See ../FROZEN_PAPER_RESULTS.md and audit/frozen_paper_results_20261112/FROZEN_RESULTS.json. Historical100fit/five-seed data remain separate.

## Verify and build

From repository root:
```sh
dissertation/tooling/venv/bin/python dissertation/scripts/verify_frozen_paper.py
dissertation/tooling/venv/bin/python dissertation/scripts/make_figures.py
./dissertation/scripts/build.sh
dissertation/tooling/venv/bin/python dissertation/scripts/count_words.py
dissertation/tooling/venv/bin/python dissertation/scripts/verify_pdf.py
```

The active generators read existing scalars only, never training/model modules, graphs or checkpoints. No randomness, inference, training, or resampling occurs. `verify_frozen_paper.py` checks all60selected raw scalars, actual public seed IDs and frozen source hashes, and generates the current table plus parameter/seed appendices. The historical verify_five_seed.py now writes coverage_historical_five_seed.tex and cannot overwrite the current table. Retired generators/assets are archived; historical audit JSON is not current verification. The PDF entry point is main.tex and the deliverable is build/dissertation_initial_draft.pdf.

Identity and official title remain in frontmatter/identity.tex. The degree is MSc in Electronic and Information Engineering; student WENTAO ZHAO (25102691G); supervisor Prof. YE Qingqing; award2027; initial submission November2026. Signature remains blank. This shortened diagnostic revision requires renewed student/supervisor review; earlier page/word counts no longer apply. No submission, public release, commit or push is performed.

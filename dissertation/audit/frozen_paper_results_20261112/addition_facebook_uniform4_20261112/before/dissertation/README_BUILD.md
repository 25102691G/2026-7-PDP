# PDP-PrivIM dissertation: fixed selected paper results

Current scope (2026-11-12): Table6.1 retains14researcher-fixed configurations, all seeds0/1/2 per cell,42fits displayed. They are selected development configurations from480fits, not independent confirmation. Other6cells are shown as dashes and all original evidence remains preserved. The14retained dataset/method cells must not be trained again or have their selected parameters/results changed. See ../FROZEN_PAPER_RESULTS.md and audit/frozen_paper_results_20261112/FROZEN_RESULTS.json. Historical100fit/five-seed data remain separate.

## Verify and build

From repository root:
```sh
dissertation/tooling/venv/bin/python dissertation/scripts/verify_frozen_paper.py
dissertation/tooling/venv/bin/python dissertation/scripts/make_figures.py
./dissertation/scripts/build.sh
dissertation/tooling/venv/bin/python dissertation/scripts/count_words.py
dissertation/tooling/venv/bin/python dissertation/scripts/verify_pdf.py
```

The active generators read existing scalars only, never training/model modules, graphs or checkpoints. No randomness, inference, training, or resampling occurs. `verify_frozen_paper.py` checks all42retained raw scalars, exact seed triplets and frozen source hashes, and generates the current table plus parameter/seed appendices. The historical verify_five_seed.py now writes coverage_historical_five_seed.tex and cannot overwrite the current table. Retired generators/assets are archived; historical audit JSON is not current verification. The PDF entry point is main.tex and the deliverable is build/dissertation_initial_draft.pdf.

Identity and official title remain in frontmatter/identity.tex. The degree is MSc in Electronic and Information Engineering; student WENTAO ZHAO (25102691G); supervisor Prof. YE Qingqing; award2027; initial submission November2026. Signature remains blank. This shortened diagnostic revision requires renewed student/supervisor review; earlier page/word counts no longer apply. No submission, public release, commit or push is performed.

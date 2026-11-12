Current manuscript reporting scope (2026-11-12): five datasets x four configurations x three runs per cell = 60 runs. Table 6.1 reports Coverage mean and sample SD. Current individual outcomes, actual public seed IDs and parameters are in the Appendix B tables; the development/source protocol and full historical cohorts are retained there separately. Frozen numerical outcomes and original provenance are unchanged. Earlier revision descriptions below are historical unless restated here.

# PDP-PrivIM dissertation: three-run Coverage comparison

## Verify and build

From repository root:
```sh
dissertation/tooling/venv/bin/python dissertation/scripts/verify_frozen_paper.py
dissertation/tooling/venv/bin/python dissertation/scripts/make_figures.py
./dissertation/scripts/build.sh
dissertation/tooling/venv/bin/python dissertation/scripts/count_words.py
dissertation/tooling/venv/bin/python dissertation/scripts/verify_pdf.py
```

The active generators read existing scalars only, never training/model modules, graphs or checkpoints. No randomness, inference, training, or resampling occurs. `verify_frozen_paper.py` checks all 60 main-table raw scalars, actual public seed IDs and frozen source hashes, and generates the current table plus parameter/seed appendices. The historical verify_five_seed.py now writes coverage_historical_five_seed.tex and cannot overwrite the current table. Retired generators/assets are archived; historical audit JSON is not current verification. The PDF entry point is main.tex and the deliverable is build/dissertation_initial_draft.pdf.

Identity and official title remain in frontmatter/identity.tex. The degree is MSc in Electronic and Information Engineering; student WENTAO ZHAO (25102691G); supervisor Prof. YE Qingqing; award2027; initial submission November2026. Signature remains blank. This shortened diagnostic revision requires renewed student/supervisor review; earlier page/word counts no longer apply. No submission, public release, commit or push is performed.

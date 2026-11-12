# PDP-PrivIM five-seed dissertation draft

Current scope (2026-11-10): all existing seeds0..4 per graph/configuration, n5/cell, 100 outcomes. Main research priority is Uniform/PDP instability; no required superiority/closeness result. Large-scale confirmation was removed from the manuscript by researcher instruction and related future experiments discontinued. Original evidence remains on disk; prior manuscript/assets are internal history under audit/revision_20261110_five_seed_stability/ and are not part of the current manuscript.

## Verify and build

From repository root:
```sh
dissertation/tooling/venv/bin/python dissertation/scripts/verify_five_seed.py
dissertation/tooling/venv/bin/python dissertation/scripts/make_figures.py
./dissertation/scripts/build.sh
dissertation/tooling/venv/bin/python dissertation/scripts/count_words.py
dissertation/tooling/venv/bin/python dissertation/scripts/verify_pdf.py
```

The active generators read existing scalars only, never training/model modules, graphs or checkpoints. No randomness, inference, training, or resampling occurs. `verify_five_seed.py` verifies every raw scalar and exact seed set and creates the current tables. Retired generators/assets are archived; historical audit JSON is not current verification. The PDF entry point is main.tex and the deliverable is build/dissertation_initial_draft.pdf.

Identity and official title remain in frontmatter/identity.tex. The degree is MSc in Electronic and Information Engineering; student WENTAO ZHAO (25102691G); supervisor Prof. YE Qingqing; award2027; initial submission November2026. Signature remains blank. This shortened diagnostic revision requires renewed student/supervisor review; earlier page/word counts no longer apply. No submission, public release, commit or push is performed.

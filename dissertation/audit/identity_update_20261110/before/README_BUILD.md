# PDP-PrivIM MSc dissertation initial draft

Prepared 2026-11-10 for internal student/supervisor review under W1. Seven complete English chapter drafts, mandatory front matter, numbered tables/figures/equations, IEEE numeric bibliography, and two appendices. Only `dissertation/` is authored. No experiment launcher is called.

## Build on this machine

From the project root:

```sh
./dissertation/scripts/build.sh
```

The script uses the local official Tectonic0.17.0 executable, its bundled BibTeX engine and a cache under `dissertation/tooling/cache`. First build needs network access for TeX packages. It preserves `build/main.log`, `build/main.blg`, `build/main.bbl` and `build/compile_console.log`, then copies the compiled PDF to `build/dissertation_initial_draft.pdf`. PDF is searchable text, not rasterized pages. `main.tex` is the manuscript entry point. `tooling/` is local tooling, not submission content.

With a separately installed TeX Live/MacTeX toolchain, an equivalent build from `dissertation/` is:

```sh
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
cp build/main.pdf build/dissertation_initial_draft.pdf
```

This alternative is documentation, not a claim that latexmk is installed here.

## Recheck existing saved data / PDF

```sh
dissertation/tooling/venv/bin/python dissertation/scripts/count_words.py
dissertation/tooling/venv/bin/python dissertation/scripts/verify_privacy_constants.py
dissertation/tooling/venv/bin/python dissertation/scripts/verify_saved_scalars.py
dissertation/tooling/venv/bin/python dissertation/scripts/verify_development.py
dissertation/tooling/venv/bin/python dissertation/scripts/make_figures.py
dissertation/tooling/venv/bin/python dissertation/scripts/verify_pdf.py
```

The verification and plotting scripts only read saved scalar observations, analysis outputs, reports and public mathematical constants. They do not import the project's training/model modules, open graph files or checkpoints, run resampling, or use a random source. Source output hashes are checked. Do not substitute historical five-seed tables for confirmation.

## Files

- `chapters/`, `frontmatter/`, `tables/`, `figures/`, `appendices/`, `references.bib`: editable manuscript sources/assets.
- `audit/`: data checks, DOI metadata, source traceability, full scalar CSVs, PDF QA and visual contact sheets; internal auxiliary material, not manuscript sections.
- `EVIDENCE_TRACEABILITY.md`, `LITERATURE_VERIFICATION.md`: claim/source map and literature review ledger.
- `DISSERTATION_REQUIREMENTS_CHECKLIST.md`, `MISSING_INFORMATION.md`, `SUBMISSION_TIMELINE.md`: compliance and human actions.
- `AI_ASSISTANCE_RECORD.md`: drafting assistance for review and any required disclosure.

Student name, official degree, award year and acknowledgements remain clear placeholders. The title is proposed, and November2026 is the draft's initial-submission setting to be confirmed before actual filing. The originality declaration is copied from the supplied school PDF with blank signature/name lines. No actual submission, Turnitin check, supervisor approval or signature has been performed.

Formatting checks are technical observations, not institutional approval. Full release privacy is not certified. No commit/push is authorized.

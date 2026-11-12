# Identity-only dissertation update — 2026-11-10

Confirmed fields are centralized in `frontmatter/identity.tex`: WENTAO ZHAO;25102691G; Master of Science in Electronic and Information Engineering (display: MSc in Electronic and Information Engineering); The Hong Kong Polytechnic University; Department of Electrical and Electronic Engineering; Prof. YE Qingqing; Associate Professor; award2027; initial submission November2026. Specialism is unconfirmed and omitted. Existing dissertation title unchanged.

Cover and title use the confirmed macros. No Supervisor field existed in the school title layout, so supervisor/position and studentID are configuration/administrative data only; no extra cover/title fields added. The originality statement is word-for-word unchanged, printed name is filled, signature line remains blank and no signing date is generated. Acknowledgements remain for the student.

## Edited files

Added `frontmatter/identity.tex`. Updated the following existing files (some generated audit/build files listed separately below):

- `WRITING_PROGRESS.md`
- `DISSERTATION_REQUIREMENTS_CHECKLIST.md`
- `FORMAT_VISUAL_REVIEW.md`
- `main.tex`
- `FINAL_DELIVERY_REPORT.md`
- `README_BUILD.md`
- `EVIDENCE_TRACEABILITY.md`
- `MISSING_INFORMATION.md`
- `frontmatter/title.tex`
- `frontmatter/cover.tex`
- `frontmatter/originality.tex`
- `audit/FINAL_STATUS.json`
- `audit/PDF_VERIFICATION.json`

Appended identity-stage notes to `progress.md` and `findings.md`; refreshed `audit/DELIVERY_SHA256.json` after preserving its earlier version under the before/ snapshot. Generated outputs include the compiled PDF, its normal build intermediates/logs, updated PDF validation/extraction/renders, and this stage's before/verification records.

## Verification

- Existing absolute `scripts/build.sh` executed successfully. Output remains `build/dissertation_initial_draft.pdf`,57pages,PDF1.5.
- Front3pages visually reviewed; only pages1–3 differ in text/render hashes. All54remaining pages are text-and-pixel identical to the pre-edit PDF.
- All7technical chapter files, appendices, tables, figures, references, abstract and saved scalarCSV files hash-identical;25protected files checked. School PDF and AGENTS unchanged.
-38,107outside-dissertation file metadata entries unchanged. No historical research files authored, no graph/checkpoint access for evaluation, no new training/inference/Coverage, no commit/push.
- AllA4; conventional1.5spacing preserved (12pt body baseline17.927pt); allpages searchable, allfonts embedded, no encryption/password.150internal links resolve,16citation destinations, no undefined references or overfull boxes.
- PDF author=WENTAO ZHAO; subject includes the official full degree. No old identity placeholders in active LaTeX sources. Historical before/ snapshots intentionally preserve the earlier state.

Technical format checks against supplied school Section4 pass. This is not institutional approval, signature, plagiarism-check completion or authorization to submit. Specialism, acknowledgements, personal contribution review and remaining supervisor/school workflow remain pending. Turnitin/EEE127/submission were not performed. Stop after this update.

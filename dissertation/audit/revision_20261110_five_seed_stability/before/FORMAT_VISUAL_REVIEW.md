# Final visual and technical review

All57 PDF pages were rendered and inspected through five contact sheets, with full-size checks of the conditional theorem/formulas, Coverage risk table, both scientific figures and reference-gap table. Chapter transitions, front matter, contents, lists, references and appendices were checked. No graphic crop, off-page text, overlapping table cell, missing visible mathematical symbol or unreadable plot legend was observed. The most recent minor revisions concerned execution-recovery wording and consistent dataset capitalization; affected pages were re-rendered.

Technical observations from `audit/PDF_VERIFICATION.json`:

- PDF1.5; all pages A4; searchable text on every page.
- No encryption dictionary/password; no PDF security restriction applied.
- Every used font has an embedded font program, including imported scientific plot fonts.
- Global conventional1.5 spacing via setspace, with measured12pt body line pitch about17.927pt. Tables/figures use permitted exceptions. Source margins30/25mm are chosen settings, not invented school requirements.
- Abstract326 words under the documented lexical rule; source and normalized PDF counts match. It fits one page.
-150 internal navigation links resolve to valid page targets;16 distinct bibliography destinations; no undefined citation/reference or overfull box in final main.log.
- Official declaration wording and examination-cover marking checked against searchable PDF text; major section order and seven chapter starts checked.

Build notes: the downloaded third-party algorithm.sty has a non-UTF8 comment that produces an input-encoding warning in Tectonic; no replacement glyph occurs in the output text or visual algorithm. One underfull paragraph warning remains and was visually checked. BibTeX main.blg completes without missing-data/style errors. These are not failed reference or clipping checks.

The QA extractor preserves default text flags and additionally dehyphenates before Unicode normalization. An earlier diagnostic that supplied only the dehyphenation flag suppressed default CID fallback for mathematical glyphs and reported43 replacement characters; that was an extraction-configuration error, not43 visible PDF defects. The corrected extraction and rendered mathematical checks are retained as the final basis.

This is a technical review against the supplied Section4 requirements, not PolyU approval. Identity fields are now researcher-confirmed; specialism remains omitted. Student signature/acknowledgements, actual contribution attribution, supervisor readiness, AI-disclosure decision, Turnitin, EEE127 and actual submission actions remain pending human work.

## Identity-only update —2026-11-10

Updated cover/title/originality pages visually inspected at full resolution. Confirmed WENTAO ZHAO, complete degree display,2027award year and November2026initial submission. Original declaration wording unchanged; typed name filled, signature/date blank. Same57pages. All54pages after the front three have identical extracted text and rendered pixel hashes to the pre-edit PDF; seven chapter sources and technical assets/data hashes unchanged. A4/spacing/searchability/embedding/security/cross-reference checks pass again. Metadata author is WENTAO ZHAO and subject includes the official full degree. Supervisor/title and studentID are recorded in identity.tex, without extra cover fields. Specialism omitted.

# PolyU / W1 compliance checklist

Authority: supplied root `Dissertation - Guidelines and Assessment.pdf` (19 pages), especially Section4 pp6–8. This is technical draft review, not school or supervisor approval. Final machine QA is in `audit/PDF_VERIFICATION.json` and visual review record. Human identity and workflow items prevent calling the document submission-ready.

| # | Requirement | Implementation / evidence | Status |
|---|---|---|---|
| 1 | §4.2 mandatory order, each major part new page | Cover, title, originality, abstract, acknowledgements, TOC, lists, Ch1–7, references, appendices; `main.tex` and PDF heading-page check | PASS; compiled order verified |
| 2 | Examination cover marking | Literal `Initial Submission for Examination Purpose` in cover | Implemented |
| 3 | No invented student/supervisor/degree/award year | Explicit uppercase placeholders; department supported by supplied school forms | PASS; actual personal information pending |
| 4 | English abstract200–500 words | `frontmatter/abstract.tex`; lexical program count326 | PASS; source count, PDF extraction also checked |
| 5 | §4.1 1.5 or double body spacing | Global `\onehalfspacing` using setspace; tables/figures allowed exceptions | PASS; measured body baseline17.927pt for12pt |
| 6 | A4 | Class option and explicit geometry, all-page MediaBox check | PASS; all57page sizes verified |
| 7 | Numbered/readable figures and tables |2 scientific figures from existing scalars; numbered LaTeX floats/tables | PASS; page renders inspected |
| 8 | Unified mathematical notation |2340 coordinates, target/bound distinction, pp vs percentages, stable U/Registry | Source review complete; student proofread pending |
| 9 | Mechanism matches archived proof | Traceability equation map; algorithm uses one outer sample, K2, doubleClip3, Δ12, fullGaussian, /16,100 updates,32-weight projection | Static source/proof review; human assumption review pending |
|10 | Correct privacy bounds | Target1 .986940326797 /2.6894142137e-6; target2/4 1.982092738992 /1e-5; Uniform1<.998694; Uniform4<3.997 | Matches trusted archived derivation/certificates; not machine certification |
|11 | Correct2900-unit statistics | Manifest hashes, all2900 rows,20 cells, primary/secondary Welch and low-tail verification | PASS; `SCALAR_VERIFICATION.json` |
|12 | Negative / inconclusive results retained |p=.1874667289, CI crosses0, no3pp support, Facebooknegative, all lowtails | PASS |
|13 | Development vs confirmation |Separate100-row historical table/CSV; seeds0..4 excluded from final estimation | PASS; `DEVELOPMENT_VERIFICATION.json` |
|14 | Relaxed Uniform4 identity |Relaxed reference, not same-policy replacement or utility upper bound | PASS |
|15 | Real literature |13 external sources;3 labeled internal archive references; DOI/URL ledger | Verified work identities; stated metadata/version limits retained |
|16 | No unsupported priority claim |No first personalized GNN or first Node-DP IM claim; standard components attributed | PASS |
|17 | Explicit limitations |Proof/release, five fixed graphs, approximation, recovery, tails, unused target4 budget | PASS |
|18 | References complete and matched |BibTeX / IEEEtran; numeric accepted under discipline alternative in §4.2.13 | PASS;16references resolved, no undefined keys |
|19 | Searchable PDF / embedded fonts / version / security |Tectonic textual PDF; pypdf/PyMuPDF inspection, font program checks | PASS; actual PDF resource/text checks |
|20 | Historical research preserved / no experiments |Only dissertation authorship; source hashes +38,106 existing-file metadata inventory; no model/graph imports | PASS; historical status and inventory unchanged |

## Other school and workflow requirements

- Certificate declaration transcribed exactly from p7; blank signature/name lines retained. Student must inspect and personally sign; no date or endorsement fabricated.
- No optional dedication/publications section is invented. Acknowledgements remain for the student.
- IEEE numbered style is a discipline convention permitted by §4.2.13; an alphabetic sort is not required when that alternative is used.
- Margins30mm left/25mm others and12pt Latin Modern are deliberate choices; school Section4 does not prescribe these exact values. Do not present them as mandatory school measurements.
- Department name is confirmed in the supplied August2026 forms. Student programme/specialism is not inferred from that name.
- Final title, year of award, actual submission month, acknowledgements and contribution attribution require confirmation.
- Turnitin or other approved electronic check before formal examination submission is required by school p19. Codex has not performed it or obtained supervisor approval.
- Timeline and EEE127/recording/finalPDF tasks are listed in SUBMISSION_TIMELINE.md; actual General Office email still needs student checking.
- AI-assisted drafting disclosure needs supervisor/Department confirmation; AI_ASSISTANCE_RECORD.md records assistance without claiming an invented policy.
- Institutional originality, dissertation acceptance, full release privacy and public submission permission are not established by this checklist.

## Final technical outcome

57-page initial draft compiled and visually reviewed. Abstract326words; chapter-prose estimate9968; fullPDF lexical count15358 includes contents/captions/references/math/numeric tokens. PDF1.5, allA4, allfonts embedded, no security, searchable on every page, no off-page text or overfull boxes,150valid internal links and16bibliography destinations. Technical Section4 checks PASS; submission readiness remains pending the personal, supervisor, integrity and administrative items above.

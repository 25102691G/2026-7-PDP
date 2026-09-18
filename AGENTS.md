# PrivIM repository instructions

## Primary specification

`PrivIM_Report.pdf` is the primary technical specification for this
repository. For changes to algorithms, privacy accounting, sampling, loss
functions, dataset preparation, experiment parameters, or evaluation:

1. Read the relevant section, algorithm, theorem, or appendix in
   `PrivIM_Report.pdf` before editing code.
2. Use `docs/paper_implementation_guide.md` as a navigation and code-mapping
   aid, not as a replacement for the paper.
3. Preserve the paper's definitions and defaults unless the user explicitly
   requests a deviation.
4. If existing behavior conflicts with the paper, do not silently treat the
   implementation as authoritative. Identify the conflict, implement the
   paper behavior when it is in scope, and document the compatibility impact.
5. In the final report, cite the paper section/algorithm/equation that guided
   any material algorithmic choice and disclose intentional deviations.

The priority order is: explicit user instruction, `PrivIM_Report.pdf`, this
file and the implementation guide, repository tests and existing code.

## Reproduction invariants

- The privacy target is unbounded node-level differential privacy.
- The paper method names map to repository settings as documented in
  `docs/paper_implementation_guide.md`; do not infer semantics from the local
  setting name alone.
- Keep privacy-sensitive quantities (`n`, `theta`, `r`, `M`, `Ng`, clipping,
  noise, sampling rate, batch size, and iteration count) explicit in code and
  output metadata.
- Algorithmic changes require focused tests for the corresponding invariant
  and, when practical, a small deterministic reproduction check.
- Evaluation comparisons must use the same prepared split, diffusion model,
  edge probability, hop count, seed-set size, and CELF denominator.

## Source integrity

The currently indexed paper has SHA-256
`cf7fc97430366bec111d230d1a945a79c844980dd791373c1d04374514c647ad`.
If the PDF changes, re-read it and update the implementation guide and this
hash before relying on the guide.

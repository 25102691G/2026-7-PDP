# W1 evidence traceability

All paths below are relative to repository root unless prefixed dissertation/. W1 writing authorization in the supplied request overrides prior PAPER_WRITING_BLOCKED for drafting only. Historical stage reports retain their original scope/status; no past authorization has been rewritten.

| Draft claim / object | Primary evidence | Verification / scope |
|---|---|---|
| School ordering, originality declaration, abstract, A4, spacing, PDF | `Dissertation - Guidelines and Assessment.pdf`, all 19 pages, especially §§4.1–4.5 pp6–8 and integrity pp17–19 | Full text extraction `audit/guidelines.txt`; all pages rendered; pp6–8 visual review. Actual filename lacks requested -1 suffix. Original untouched. |
| Department formal name | School forms pp9,14,16 | EEE confirmed; does not establish student's degree/specialism or award year. |
| IM, GNN and PrivIM background | Root `PrivIM_Report.pdf`, model/source; primary external literature | Local 18-page paper extracted, references and method sections read. Bibliographic verification ledger separate. |
| Chapter 3 adjacency/domain | `docs/pdp_research/G0_THEORY_BOUNDARY_REVIEW.md` §§2,8; `PREPARED_DOMAIN_PROTOCOL.md`; U5 proof readiness | Fixed U/Registry, surviving isolated nodes and stable identities; not arbitrary file-row deletion. |
| Chapter 4 architecture and loss | `model/GRAT.py`, `model/layer/GRATLayer.py`, `model/functional/MaxCoverLoss.py`, `experiments/graph_util.py` | Static reads only; legacy released model, all-one32, A+I, loss .5; dimension 2340 checked algebraically. |
| Slots, cached samplers and clipping | `experiments/node_dp_slots/{config,adapter,aggregation,mechanism,training}.py` | Four slots, K2, four padded positions, fixed /16, slot-local caches; clip offset/margin disclosed. |
| Bernoulli and projection | `experiments/node_pdp/{registry,sampling,accounting}.py`; `experiments/node_pdp_v2_projected/policy.py` | One outer sample; p1=.268941..., p2=p4=1; 32 weights only, no initial/bias projection. |
| Conditional proof chain | G0 §§4–7; U0 `PRIVACY_PROOF.md`; U2 `PRIVACY_PROOF.md`, certificates | Sensitivity12; Gaussian full2340; adaptive100; grid RDP conversion; two-direction amplification. Not machine certification. |
| Uniform1 bound | `experiments/uniform_node_dp/accounting.py`; U0 report/proof/certificate | sigma486, epsilon numeric .9986929644, conservative <.998694. |
| Uniform4 bound | `docs/pdp_research/stage_u2_uniform_eps4_mechanism/{PRIVACY_PROOF.md,accounting_certificate.json}` | sigma139, epsilon numeric3.99699597485, strict upper3.997. Relaxed reference only. |
| NP route and common parameters | U6 `FROZEN_MECHANISM_AUDIT.md`, `FROZEN_PROTOCOL.json`; `experiments/train_models.py`, original training source | Ordinary SGD, no private clipping/slots/projection. NP not a pure noise ablation. |
| Domain and node counts | U3 `PREFLIGHT_DATA_AUDIT.json`; U4 `PUBLIC_POLICY_COUNTS.json` | Already saved metadata, no graph files read. Email U1005 versus active train462/test458. |
| Top50 / IC / ranking | `experiments/pdp_research_evaluation.py`; frozen SAP | Pre-sigmoid, original stable local-ID ties; one-step outgoing IC p1; denominator432/1684/1269/1741/3157. |
| 2900 final observations | `docs/pdp_research/stage_u6a_posttraining_analysis/runtime/outputs/{observations.json,analysis.json,REPORT.md,OUTPUT_MANIFEST.json}` | Output manifest hashes verified; full independent deterministic arithmetic verification in `audit/SCALAR_VERIFICATION.json`. |
| Statistical formulas, weighting, sample sizes | Original U6-F `FROZEN_STATISTICAL_ANALYSIS_PLAN.md`, byte-preserved U6-A plan; `ua_statistics.py` read only | Ten cells for primary variance; no t4, no graph pooling, no paired assumptions; n200/200/140/40. |
| Recovery and attempts | U6-R2 `APPROVED_AMENDMENT.md`; U6-A `evidence.json`, `PARENT_SNAPSHOT.json`; U6-R2 posttraining `DIAGNOSIS.json` | 736 reused+2164 new=2900; 2902 attempts;2 failed fits; two separately authorized same-public-seed technical second attempts (Email entry path and Facebook timeout); later analysis wall guard preserved separately. |
| Primary and secondary figures/tables | Final U6-A saved observations and analysis | `scripts/verify_saved_scalars.py`, `scripts/make_figures.py`; deterministic only, no graph or model imports, no new resampling. |
| Sensitivity intervals | Sealed `analysis.json` bootstrap and Hoeffding fields | Copied with role sensitivity only; 10000 stored resamples never rerun in W1. |
| Development100 | U5 `PER_SEED_RECHECK.csv`; U4 `GROUP_STATISTICS.csv`; U3 report | Complete100-row CSV retained; independently rechecked from Spread/CELF. Excluded from confirmation. |
| New Matched historical | `docs/pdp_research/new_matched_pdp_development/REPORT.md`, `outputs/research/new-matched-frozen-dev-20261103/restricted/summary.csv`; R0 report/fairness audit | 2307 frozen/33 trained, n5; four favorable means and adverse HepPh result all retained. |
| G2 Gaussian diagnostic | U4 `G2_INDEPENDENT_SCALAR_RECHECK.json`, `G2_CONTROLLED_PROTOCOL.md` | Historical witness-controlled path differences only; no complete variance decomposition. |
| G3/C3 failure | `G3_AVERAGE_PROTOCOL.md`, `G3_AVERAGE_THEORY.md`, `outputs/research/g3-parameter-average-20261102-v1/restricted/analysis.json` and G3_REPORT.md; `stage_c3_rowsum/REPORT.md` | Rejected development candidates; not final algorithm or independent confirmation. |
| U0–U6 decisions | Stage reports U0/U1/U2/U3/U4/U5/U6P/U6D/U6F and amendments | Chronology separated, no archived report rewritten. Later decision supersedes earlier prospective missing-data status. |

## Proof-to-equation map

| LaTeX label | Trusted mathematical source |
|---|---|
| eq:doubleclip, lem:sensitivity | G0 §§3–4; U2 §2 |
| eq:update | G0 §3; U2 §§3–4; policy.step |
| eq:gaussrdp, eq:rdp | G0 §5; U2 §§3–4; Mironov for RDP framework |
| eq:conversion | G0 §6; U2 §5; not substituted by a different textbook conversion |
| eq:participation, eq:amplified, thm:pdp | G0 §7; node_pdp accounting/Registry |
| eq:welch and test formulas | U6 frozen SAP §§1–4; saved ua_statistics.py |

Human review remains required for application of ideal theorem assumptions and intended disclosure scope. No proof of raw preparation, private test release, all-model joint privacy or full machine implementation is invented. Those extensions are PROOF_REVIEW_REQUIRED.

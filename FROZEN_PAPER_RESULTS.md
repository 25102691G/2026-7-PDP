# Frozen paper results — 2026-11-12

Researcher instruction: all20dataset/method cells are selected for dissertation Table6.1 (60displayedfits). Original15cells retain their complete seeds0/1/2; five added cells are explicitly post hoc three-run subsets with their actual public initialization IDs. Do not retrain or change selected outcomes/settings. All original results and full source-cohort summaries remain preserved.

- email / nonprivate: n=40, M=4, T=500; 96.451 +/- 2.100; seeds0,1,2.
- email / uniform4: n=20, M=4, T=100; 97.608 +/- 1.541; seeds0,1,2.
- bitcoin / nonprivate: n=40, M=2, T=100; 99.070 +/- 0.386; seeds0,1,2.
- bitcoin / pdp_v2: n=60, M=4, T=100; 96.952 +/- 1.734; seeds0,1,2.
- bitcoin / uniform4: n=20, M=4, T=100; 97.387 +/- 0.992; seeds0,1,2.
- lastfm / nonprivate: n=40, M=4, T=300; 96.270 +/- 1.343; seeds0,1,2.
- lastfm / pdp_v2: n=40, M=4, T=300; 95.640 +/- 1.058; seeds0,1,2.
- lastfm / uniform4: n=40, M=4, T=500; 95.088 +/- 0.298; seeds0,1,2.
- hepph / uniform1: n=40, M=4, T=500; 83.649 +/- 3.219; seeds0,1,2.
- facebook / pdp_v2: n=40, M=2, T=100; 91.395 +/- 1.843; seeds0,1,2.

- facebook / nonprivate: n=60, M=2, T=300; 95.238 +/- 3.150; seeds0,1,2.
- email / uniform1: n=40, M=8, T=100; 85.417 +/- 1.518; seeds0,1,2.
- bitcoin / uniform1: n=40, M=2, T=100; 82.324 +/- 3.734; seeds0,1,2.
- email / pdp_v2: n=40, M=4, T=500; 90.355 +/- 4.045; seeds0,1,2.

- facebook / uniform4: n=80, M=2, T=100; 95.650 +/- 2.144; seeds0,1,2; source: six-cell-followup-20261112-v1.

- lastfm / uniform1: n=40, M=4, T=100; 80.877 +/- 0.164; public seeds=[647226721, 1235008776, 1587065786]; POST_HOC_SUBSET of N=200.
- facebook / uniform1: n=40, M=4, T=100; 82.589 +/- 0.439; public seeds=[342613339, 1098033150, 1721636732]; POST_HOC_SUBSET of N=200.
- hepph / pdp_v2: n=40, M=4, T=100; 93.509 +/- 0.000; public seeds=[671557794, 1039941809, 1273303461]; POST_HOC_SUBSET of N=200.
- hepph / uniform4: n=40, M=4, T=100; 95.788 +/- 0.120; public seeds=[525706892, 1463775085, 2067563573]; POST_HOC_SUBSET of N=140.
- hepph / nonprivate: n=40, M=4, T=100; 94.945 +/- 0.470; public seeds=[280195270, 498909545, 1911729556]; POST_HOC_SUBSET of N=40.

Authoritative mapping/hashes: dissertation/audit/frozen_paper_results_20261112/FROZEN_RESULTS.json. Full source cohorts: U6_FULL_SOURCE_COHORTS.json in the same directory. Generator/verification: dissertation/scripts/verify_frozen_paper.py. These20selected cells do not constitute unselected independent confirmation. HepPh Non-Private94.945 remains below95; selected PDP SD0 does not imply population variance0.

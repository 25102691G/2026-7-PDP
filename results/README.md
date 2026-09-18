# Curated experiment records

This directory documents the small, durable subset of generated experiment
outputs retained in Git. The source files remain under their original
`outputs/runs/<run-id>/` paths so commands in manifests and the handoff remain
traceable.

Tracked run records contain only `manifest.json` and
`evaluation/summary.csv` for the latest controlled feature experiments:

- `diagnostic-facebook-degree32`
- `diagnostic-email-degree32`
- `diagnostic-bitcoin-lastfm-degree32`
- `diagnostic-bitcoin-lastfm-ones-controlled`

Model checkpoints, per-seed model metadata, and per-seed evaluation CSV files
remain ignored. Copy or archive the complete `outputs/runs/` directory when
checkpoint-level continuation is required.

The exact prepared splits under `dataset/prepared/` are tracked separately.
Raw datasets are not tracked; verify them with
`raw_dataset/checksums.sha256`.

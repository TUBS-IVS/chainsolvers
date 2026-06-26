# Block A audit — frozen canonical numbers

This directory is the **tracked, durable record** of every quantitative Block A claim in the paper,
so paper prose and data can never silently drift apart again (which is exactly what happened when the
worlds were re-baked and only some results were re-run).

`research/out/` and `research/data/` are gitignored (disposable generated output / huge+licensed MiD),
so the working CSVs and baked worlds are **not** tracked. This dir freezes the small, derived,
MiD-free artifacts that pin the paper.

## Contents

- **`AUDIT.txt`** — every paper-cited Block A number, per world: gap-to-oracle × difficulty (R1),
  proven-optimality %, generation/search decomposition (R4), recall geometry-limit floor (R6),
  chain-length scaling (R3), the N-wall (R5), the quality–runtime frontier ranges (R2), and the
  anchor-quality subpopulation degradation. This is the source of truth — paper numbers must match it.
- **`<world>_anchor_subpop_raw.csv`** — raw per-person deviations (true vs anchor-disturbed) for the
  genuinely work-bounded subpopulation, the evidence behind the "RDA degrades +240/+554 m on the city
  worlds while exact DP stays at the re-optimized oracle" claim. (See the `block-a-anchor-axis-metric`
  memory: measure as gap-above-oracle on the ~5–8% affected subpop, NOT all-persons.)
- **`<world>_dp_sample_tuned_calib.json`** — the cached, deterministic `dp_sample_tuned` calibration
  (MLE body scale + mixture tail) used for the A1b generative reference.

## Regenerate

```
python research/scripts/block_a_audit.py
```

Reads the working result CSVs under `research/out/block_a/<world>/` and reconstructs the affected
subpopulation deterministically (seed 0, same rng stream as `result_gap_difficulty`). The numbers here
correspond to the worlds baked on 2026-06-25 (gauss/osm content-stable re-saves 15:24/15:29;
two_zone re-bake 18:38 — `busi` separate, MiD templates min 10 respondents). **If a world is re-baked,
re-run `block_a.py` for that world, re-run this audit, and re-check the paper against the new
`AUDIT.txt`** — a re-bake can move the numbers (e.g. the two_zone generation gap was 3.7 m before the
18:38 re-bake, 2.4 m after).

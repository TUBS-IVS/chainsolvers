# Comparing against eqasim (vendored, private)

eqasim is **GPL-3.0**; chainsolvers is **MIT**. GPL copyleft triggers only on *distribution*, so you
may **vendor eqasim and import it for your own (non-distributed) experiments** — as long as it never
ships with the MIT library. All of it stays under gitignored paths (`vendor/`, `_private/`): never
committed, packaged, or imported from `chainsolvers/`. Reporting *results* in a paper is fine; to let
others reproduce the harness, release it as a separate GPL repo.

> **This is a scaffold, not running code.** Nothing here runs until you vendor eqasim and wire the
> glue to *your* vendored version's API. The signatures below are illustrative — verify them.

## Two different RDAs live in two different repos

- **Pure RDA** (Hörl & Axhausen 2023): `eqasim-org/ile-de-france`, under
  `synthesis/population/spatial/secondary/` (`rda.py`, `problems.py`, `locations.py`).
- **Guidance-force RDA** (Langrognet, Côme, Hörl & Oukhellou 2026): per the paper's footnote, the
  reference implementation is in **`eqasim-org/eqasim-france`, PR #385** — *not* ile-de-france.
  (A `force_model/` also appears on `ile-de-france@develop`; confirm which matches the paper before
  using it.)

So you may need **two clones**.

## 1. Vendor (gitignored)

```bash
mkdir -p vendor
git clone https://github.com/eqasim-org/ile-de-france  vendor/ile-de-france        # pure RDA
git clone https://github.com/eqasim-org/eqasim-france  vendor/eqasim-france        # guidance RDA
#   then check out the PR #385 branch/commit in vendor/eqasim-france for the guidance code
# (pin commits/tags for reproducibility)
```

## 2. What you are (and are NOT) running

You call eqasim's **relaxation solver core** (`GravityChainSolver`) directly on *our* per-leg
distances. You deliberately **bypass** eqasim's `FeasibleDistanceSampler` + `AssignmentSolver`, which
would (a) sample target distances from a distribution and (b) loop with keep-best/feasibility. So:

- This is the **real eqasim relaxation on fixed distances** — apt for a *same-distances per-instance*
  comparison against `carla`/`dp_refine` (identical endpoints, distances, candidate set).
- It is **not the full RDA algorithm**. For that (its resample → relax → discretize → keep-best loop)
  drive `AssignmentSolver`; but then it samples its *own* distances → that is the **distributional**
  comparison (also obtainable via the arms-length `eval/interop.py` CSV path), not same-distances.

## 3. Glue (edit to your vendored version's API)

See `_private/eqasim_glue.py`. Key correctness points it must get right:

- **Thread the harness `rng`** into the solver — `GravityChainSolver` uses randomness (lateral init +
  the one-sided angular solution), and its constructor takes a random source. Without this the run is
  unseeded and not aligned with the harness `rng_seed`. (Mind RandomState vs numpy Generator.)
- **Build the full problem.** The solver needs the chain: origin `S`, destination `E`, the **per-leg
  target distances**, and the **activity count** (n−1 free nodes) — not just origin/destination.
  In eqasim the distances are typically carried *inside* the problem object; verify whether `solve`
  takes them as a second argument or reads them from the problem.
- **Honour `result["valid"]`.** Decide a policy for non-converged relaxations (accept best-effort,
  resample, or drop) rather than silently using whatever came back.
- **Discretize to OUR candidates** (below) so the candidate set is shared.

## 4. Run through the normal pipeline

```python
import sys, numpy as np
from chainsolvers import run
from chainsolvers.eval.external import CallableSolver
from chainsolvers.eval.synth import generate_world
sys.path.insert(0, "_private"); from eqasim_glue import eqasim_place

w = generate_world(rng=np.random.default_rng(0))
ctx = run.setup(locations_tuple=w.locations_tuple, solver=CallableSolver, rng_seed=1,
                parameters={"place_fn": eqasim_place})
rdf, _, valid = run.solve(ctx=ctx, plans_df=w.plans_df)
# score rdf with the same metrics used for carla / dp_refine / ... (same-distances comparison)
```

The in-library `eval/baselines.py` reimplementations remain the MIT-clean, dependency-free option for
shipped experiments; use this vendored path when you want the genuine eqasim relaxation in the loop.

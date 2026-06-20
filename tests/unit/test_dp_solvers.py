"""Tests for the exact DP / MILP solvers and their relationship to CARLA.

Covers:
- `solve_chain` (DP) returns the exact optimum vs brute-force enumeration.
- MILP oracle returns the same optimum as DP over identical candidate pools.
- End-to-end parity: `dp` == `carla_dp` for chains longer than two legs
  (they differ only in single-intermediate candidate generation).
- End-to-end validity for `dp`, `carla_dp`, `milp` through setup/solve.
- DP is never worse than CARLA on total distance deviation (search is exact).
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from chainsolvers import run
from chainsolvers.solvers.dp import solve_chain


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _objective(S, E, distances, chosen, alpha, beta):
    """alpha*sum(P) - beta*sum(|d_obs - ||leg|||) for a chosen assignment."""
    coords = [S] + [c[1] for c in chosen] + [E]
    pots = [0.0] + [c[2] for c in chosen] + [0.0]
    val = 0.0
    for i in range(1, len(coords)):
        val -= beta * abs(distances[i - 1] - np.linalg.norm(coords[i] - coords[i - 1]))
    val += alpha * sum(pots)
    return val


def _brute_force(S, E, distances, pools, alpha, beta):
    """Exhaustively maximize the objective over all candidate combinations."""
    best_val = -np.inf
    best = None
    for combo in itertools.product(*[range(len(p[0])) for p in pools]):
        chosen = [(pools[j][0][combo[j]], pools[j][1][combo[j]], float(pools[j][2][combo[j]]))
                  for j in range(len(pools))]
        val = _objective(S, E, distances, chosen, alpha, beta)
        if val > best_val:
            best_val, best = val, chosen
    return best_val, best


def _make_pools(rng, num_free, k, box=40.0, with_pots=False):
    pools = []
    for j in range(num_free):
        coords = rng.uniform(-box, box, size=(k, 2))
        ids = np.array([f"n{j}_{i}" for i in range(k)], dtype=object)
        pots = rng.uniform(0, 5, size=k) if with_pots else np.ones(k)
        pools.append((ids, coords, pots))
    return pools


def _grid_locations(types, lo=-10.0, hi=60.0, step=5.0):
    xs = np.arange(lo, hi + 1e-9, step)
    ys = np.arange(-30.0, 30.0 + 1e-9, step)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=float)
    ids = {t: np.array([f"{t}_{i}" for i in range(len(grid))], dtype=object) for t in types}
    coords = {t: grid.copy() for t in types}
    pots = {t: np.ones(len(grid)) for t in types}
    return ids, coords, pots


def _chain_df(act_types, distances, start=(0.0, 0.0), end=(50.0, 0.0)):
    """One person, len(distances) legs; intermediate to-locations unknown."""
    n = len(distances)
    rows = []
    for k in range(n):
        rows.append({
            "unique_person_id": "p",
            "unique_leg_id": f"l{k}",
            "to_act_type": act_types[k],
            "distance_meters": float(distances[k]),
            "from_x": start[0] if k == 0 else np.nan,
            "from_y": start[1] if k == 0 else np.nan,
            "to_x": end[0] if k == n - 1 else np.nan,
            "to_y": end[1] if k == n - 1 else np.nan,
        })
    return pd.DataFrame(rows)


def _total_dev(rdf):
    actual = np.hypot(rdf["to_x"] - rdf["from_x"], rdf["to_y"] - rdf["from_y"])
    return float(np.abs(rdf["distance_meters"] - actual).sum())


# --------------------------------------------------------------------------- #
# unit: DP / MILP exactness on raw pools
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("with_pots", [False, True])
def test_dp_matches_brute_force(seed, with_pots):
    rng = np.random.default_rng(seed)
    S = np.array([0.0, 0.0]); E = np.array([30.0, 5.0])
    num_free = 3
    distances = rng.uniform(8, 20, size=num_free + 1)
    pools = _make_pools(rng, num_free, k=5, with_pots=with_pots)
    alpha, beta = (1.0, 1.0) if with_pots else (0.0, 1.0)

    chosen = solve_chain(S, E, distances, pools, alpha, beta, method="dp")
    dp_val = _objective(S, E, distances, chosen, alpha, beta)
    bf_val, _ = _brute_force(S, E, distances, pools, alpha, beta)
    assert dp_val == pytest.approx(bf_val, abs=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_milp_matches_dp(seed):
    rng = np.random.default_rng(100 + seed)
    S = np.array([0.0, 0.0]); E = np.array([25.0, -4.0])
    num_free = 3
    distances = rng.uniform(8, 18, size=num_free + 1)
    pools = _make_pools(rng, num_free, k=5, with_pots=True)
    alpha, beta = 0.7, 1.3

    dp_val = _objective(S, E, distances, solve_chain(S, E, distances, pools, alpha, beta, "dp"), alpha, beta)
    milp_val = _objective(S, E, distances, solve_chain(S, E, distances, pools, alpha, beta, "milp"), alpha, beta)
    assert milp_val == pytest.approx(dp_val, abs=1e-6)


def test_single_free_node_picks_best():
    # n=2: one intermediate; DP reduces to argmax over the pool.
    S = np.array([0.0, 0.0]); E = np.array([10.0, 0.0])
    distances = np.array([5.0, 5.0])
    coords = np.array([[5.0, 0.0], [5.0, 3.0], [0.0, 0.0]])  # (5,0) is exact intersection
    pools = [(np.array(["a", "b", "c"], dtype=object), coords, np.ones(3))]
    chosen = solve_chain(S, E, distances, pools, 0.0, 1.0, "dp")
    assert chosen[0][0] == "a"


# --------------------------------------------------------------------------- #
# end-to-end through setup/solve
# --------------------------------------------------------------------------- #

def test_dp_carla_dp_milp_run_and_valid():
    types = ["work", "shop", "leisure"]
    loc = _grid_locations(types)
    df = _chain_df(["work", "shop", "leisure", "hotel"], [20.0, 15.0, 15.0, 20.0])
    for solver in ["dp", "carla_dp", "dp_refine", "carla_dp_refine", "milp"]:
        ctx = run.setup(locations_tuple=loc, solver=solver, rng_seed=42)
        rdf, plans, valid = run.solve(ctx=ctx, plans_df=df)
        assert valid, f"{solver} produced invalid output"
        assert len(rdf) == 4


def test_dp_equals_carla_dp_for_long_chains():
    # They differ only in the single-intermediate (2-leg) generation, so for
    # chains with >= 2 free nodes the placements must coincide.
    types = ["work", "shop", "leisure"]
    loc = _grid_locations(types)
    df = _chain_df(["work", "shop", "leisure", "hotel"], [18.0, 14.0, 16.0, 22.0])

    out = {}
    for solver in ["dp", "carla_dp"]:
        ctx = run.setup(locations_tuple=loc, solver=solver, rng_seed=42)
        rdf, _, _ = run.solve(ctx=ctx, plans_df=df)
        out[solver] = rdf.sort_values("unique_leg_id").reset_index(drop=True)

    for col in ["to_x", "to_y", "to_act_identifier"]:
        a, b = out["dp"][col].to_numpy(), out["carla_dp"][col].to_numpy()
        # The END anchor leg has no identifier (NaN); NaN == NaN is False, so
        # treat positions that are NaN in both as equal.
        assert ((a == b) | (pd.isna(a) & pd.isna(b))).all(), col


def test_dp_not_worse_than_carla():
    # Exact search cannot lose to heuristic branching when generation is comparable.
    rng = np.random.default_rng(7)
    n_legs = 6
    types = [f"a{j}" for j in range(n_legs - 1)] + ["END"]
    facility_types = types[:-1]
    coords = {t: rng.uniform(-60, 60, size=(400, 2)) for t in facility_types}
    ids = {t: np.array([f"{t}_{i}" for i in range(400)], dtype=object) for t in facility_types}
    pots = {t: np.ones(400) for t in facility_types}
    df = _chain_df(types, rng.uniform(12, 28, size=n_legs).tolist())

    devs = {}
    for solver in ["carla", "dp", "carla_dp"]:
        ctx = run.setup(locations_tuple=(ids, coords, pots), solver=solver, rng_seed=42)
        rdf, _, valid = run.solve(ctx=ctx, plans_df=df)
        assert valid
        devs[solver] = _total_dev(rdf)

    # DP is exact over its candidate set; allow a tiny tolerance for float noise.
    assert devs["dp"] <= devs["carla"] + 1e-6
    assert devs["dp"] == pytest.approx(devs["carla_dp"], abs=1e-6)


def test_dp_refine_never_worse_than_dp():
    # Iterative refinement starts from the one-shot DP solution and carries the
    # previous choice forward, so it is monotone: never worse than `dp`.
    rng = np.random.default_rng(11)
    n_legs = 7
    types = [f"a{j}" for j in range(n_legs - 1)] + ["END"]
    facility_types = types[:-1]
    coords = {t: rng.uniform(-60, 60, size=(300, 2)) for t in facility_types}
    ids = {t: np.array([f"{t}_{i}" for i in range(300)], dtype=object) for t in facility_types}
    pots = {t: np.ones(300) for t in facility_types}
    df = _chain_df(types, rng.uniform(12, 28, size=n_legs).tolist())

    devs = {}
    for solver in ["dp", "dp_refine"]:
        ctx = run.setup(locations_tuple=(ids, coords, pots), solver=solver, rng_seed=42)
        rdf, _, valid = run.solve(ctx=ctx, plans_df=df)
        assert valid
        devs[solver] = _total_dev(rdf)
    assert devs["dp_refine"] <= devs["dp"] + 1e-6


def test_dp_full_is_global_lower_bound():
    # dp_full optimizes over the full candidate set -> true global optimum on the
    # separable objective. No other solver can have lower total deviation.
    rng = np.random.default_rng(13)
    n_legs = 5
    types = [f"a{j}" for j in range(n_legs - 1)] + ["END"]
    facility_types = types[:-1]
    coords = {t: rng.uniform(-50, 50, size=(120, 2)) for t in facility_types}
    ids = {t: np.array([f"{t}_{i}" for i in range(120)], dtype=object) for t in facility_types}
    pots = {t: np.ones(120) for t in facility_types}
    df = _chain_df(types, rng.uniform(12, 26, size=n_legs).tolist())

    devs = {}
    for solver in ["carla", "dp", "carla_dp", "dp_refine", "dp_full"]:
        ctx = run.setup(locations_tuple=(ids, coords, pots), solver=solver, rng_seed=42,
                        parameters={"min_candidates": 8} if solver in {"dp", "carla_dp", "dp_refine"} else None)
        rdf, _, valid = run.solve(ctx=ctx, plans_df=df)
        assert valid
        devs[solver] = _total_dev(rdf)

    g = devs["dp_full"]
    for s in ["carla", "dp", "carla_dp", "dp_refine"]:
        assert devs[s] >= g - 1e-6, f"{s}={devs[s]} beat the global optimum {g}"
    # refinement should be no worse than one-shot dp and no better than global
    assert g - 1e-6 <= devs["dp_refine"] <= devs["dp"] + 1e-6

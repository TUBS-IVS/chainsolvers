"""Bake the three canonical evaluation worlds to fixed, reloadable snapshots.

Each world is built at its **full mobile population** (so potentials are a realistic dense usage
field), written via `worlds.save_world` (parquet), and rendered to a `world.png` from the **full
baked population** (the figure depicts the real world, not a sample). A `manifest.json` records the
exact recipe — builder, kwargs, seed, git SHA, row counts, file md5s — so every snapshot is
reproducible: same recipe + seed => byte-identical world.

    uv run --project research python research/scripts/bake_worlds.py            # all three, full pop
    uv run --project research python research/scripts/bake_worlds.py --only gauss_hannover
    uv run --project research python research/scripts/bake_worlds.py --quick 2000   # tiny test build

Outputs to research/data/worlds/<name>/ (gitignored). The OSM world is skipped if its cached
snapshots (research/data/hannover_pois.csv, hannover_homes.csv) are absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time

import numpy as np

from chainsolvers_eval.synth import city_world, two_zone_world
from chainsolvers_eval.worlds import save_world

HERE = os.path.dirname(__file__)
OUT_ROOT = os.path.join(HERE, "..", "data", "worlds")

# The canonical recipes. `builder` dispatches to a function below; `kwargs` + `seed` fully
# determine the world (verified deterministic in tests). Bump a seed only to intentionally
# re-roll a snapshot. PNG is rendered from `png_persons` sampled persons at `png_seed`.
RECIPES = {
    "gauss_hannover": dict(builder="city_world", kwargs={"name": "hannover"}, seed=20260623),
    "two_zone":       dict(builder="two_zone_world", kwargs={"heavy_tail": True}, seed=20260623),
    "osm_hannover":   dict(builder="hannover_osm_world", kwargs={}, seed=20260623),
}


def _build(builder: str, kwargs: dict, rng):
    if builder == "city_world":
        return city_world(rng=rng, **kwargs)
    if builder == "two_zone_world":
        return two_zone_world(rng=rng, **kwargs)
    if builder == "hannover_osm_world":
        from chainsolvers_eval.osm import hannover_osm_world
        return hannover_osm_world(rng=rng, **kwargs)
    raise KeyError(f"unknown builder {builder!r}")


def _env() -> dict:
    """Record the toolchain so a future re-bake knows what to pin (RNG stream stability is the
    main 5-year reproducibility risk — numpy reserves the right to change Generator methods)."""
    import platform
    import numpy, pandas, pyarrow
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "pyarrow": pyarrow.__version__}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_png(world, path: str):
    """Render the 4-panel overview from the **full baked population** (topology, realized usage,
    and free-leg distribution all reflect the real world). Best-effort: a viz/matplotlib failure
    won't fail a bake."""
    from chainsolvers_eval.viz import plot_world
    plot_world(world, path)


def bake_one(name: str, quick: int | None, out_root: str) -> dict:
    rec = RECIPES[name]
    kwargs = dict(rec["kwargs"])
    if quick:
        kwargs["n_persons"] = quick
    out_dir = os.path.join(out_root, name)
    print(f"[{name}] building ({rec['builder']}, kwargs={kwargs}, seed={rec['seed']}) ...", flush=True)
    t = time.time()
    world = _build(rec["builder"], kwargs, np.random.default_rng(rec["seed"]))
    build_s = time.time() - t
    m = world.meta
    print(f"[{name}] built {m['n_persons']:,} persons / {m['n_legs']:,} legs in {build_s:.0f}s; "
          f"saving ...", flush=True)
    save_world(world, out_dir)
    files = {fn: _md5(os.path.join(out_dir, fn)) for fn in
             ("facilities.parquet", "plans.parquet", "ground_truth.parquet", "meta.json")}
    # realized usage field (the potentials handed to the solvers): density check of the bake
    allp = np.concatenate([np.asarray(world.locations_tuple[2][t], float) for t in m["types"]])
    v = allp[allp > 0]
    potentials = {"visited_frac": round(float((allp > 0).mean()), 4),
                  "median_visits": float(np.median(v)) if v.size else 0.0,
                  "p75_visits": float(np.percentile(v, 75)) if v.size else 0.0,
                  "p95_visits": float(np.percentile(v, 95)) if v.size else 0.0,
                  "mean_visits": round(float(v.mean()), 2) if v.size else 0.0,
                  "max_visits": float(allp.max()) if allp.size else 0.0}
    png_ok = True
    try:
        _render_png(world, os.path.join(out_dir, "world.png"))
    except Exception as e:  # pragma: no cover - viz is optional
        png_ok = False
        print(f"[{name}] PNG render skipped: {e}", flush=True)
    return {
        "builder": rec["builder"], "kwargs": kwargs, "seed": rec["seed"],
        "n_persons": int(m["n_persons"]), "n_facilities": int(m["n_locations"]),
        "n_legs": int(m["n_legs"]), "n_free_legs": int(m["n_free_legs"]),
        "potentials": potentials,
        "build_seconds": round(build_s, 1), "files": files, "png": png_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(RECIPES), help="bake just this world")
    ap.add_argument("--quick", type=int, default=None,
                    help="override n_persons with a small count for a fast test bake")
    ap.add_argument("--out", default=OUT_ROOT, help="output root dir")
    args = ap.parse_args()
    out_root = args.out

    names = [args.only] if args.only else list(RECIPES)
    # Merge into an existing manifest so `--only` updates one world without dropping the others'
    # entries (a full run still rewrites every world it bakes). git/env/quick reflect this run.
    manifest = {"git_sha": _git_sha(), "env": _env(), "png_render": "full_population",
                "quick": args.quick, "worlds": {}}
    mpath = os.path.join(out_root, "manifest.json")
    if args.only and os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            manifest["worlds"] = json.load(f).get("worlds", {})
    for name in names:
        if name == "osm_hannover":
            from chainsolvers_eval.osm import DEFAULT_HANNOVER_POIS
            if not os.path.exists(DEFAULT_HANNOVER_POIS):
                print(f"[{name}] skipped: {DEFAULT_HANNOVER_POIS} absent", flush=True)
                continue
        manifest["worlds"][name] = bake_one(name, args.quick, out_root)

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {os.path.join(out_root, 'manifest.json')}", flush=True)
    for n, w in manifest["worlds"].items():
        print(f"  {n}: {w['n_persons']:,} persons, {w['n_legs']:,} legs, png={w['png']}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate a synthetic ground-truth world and render it to a PNG.

    python research/scripts/visualize_world.py --persons 500 --noise 0.0 --out research/out/world.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from chainsolvers_eval.synth import CITY_PRESETS, city_world, generate_world
from chainsolvers_eval.viz import plot_world


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", choices=sorted(CITY_PRESETS),
                   help="use a real-city preset (calibrated density/footprint/trip lengths); "
                        "ignores --locations/--gravity-scale")
    p.add_argument("--locations", type=int, default=1000)
    p.add_argument("--persons", type=int, default=500)
    p.add_argument("--gravity-scale", type=float, default=4000.0)
    p.add_argument("--noise", type=float, default=0.0, help="relative noise on observed leg distances")
    p.add_argument("--n-chains", type=int, default=8, help="example chains to draw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="research/out/world.png")
    args = p.parse_args(argv)

    if args.city:
        world = city_world(args.city, n_persons=args.persons, distance_noise=args.noise,
                           rng=np.random.default_rng(args.seed))
    else:
        world = generate_world(n_locations=args.locations, n_persons=args.persons,
                               gravity_scale=args.gravity_scale, distance_noise=args.noise,
                               rng=np.random.default_rng(args.seed))
    path = plot_world(world, args.out, n_chains=args.n_chains, seed=args.seed)
    m = world.meta
    print(f"Wrote {path}  ({m['n_persons']} persons, {m['n_legs']} legs, "
          f"{m['n_free_legs']} to place, {len(world.topology.types)} types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

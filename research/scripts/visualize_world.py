#!/usr/bin/env python
"""Generate a synthetic ground-truth world and render it to a PNG.

    python research/scripts/visualize_world.py --persons 500 --noise 0.0 --out research/out/world.png
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from chainsolvers_eval.synth import CITY_PRESETS, city_world, generate_world, two_zone_world
from chainsolvers_eval.viz import plot_world


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world", choices=["generic", "city", "osm", "two-zone"], default="generic",
                   help="generic gauss | OSM-calibrated gauss city | real-OSM geometry | two-zone")
    p.add_argument("--city", default="hannover", help="city preset name (for --world city)")
    p.add_argument("--locations", type=int, default=1000)
    p.add_argument("--persons", type=int, default=500)
    p.add_argument("--gravity-scale", type=float, default=4000.0)
    p.add_argument("--noise", type=float, default=0.0, help="relative noise on observed leg distances")
    p.add_argument("--n-chains", type=int, default=8, help="example chains to draw")
    p.add_argument("--heavy-tail", action="store_true", help="two-zone: 120 km long-tail super-region")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="research/out/world.png")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    if args.world == "city":
        world = city_world(args.city, n_persons=args.persons, distance_noise=args.noise, rng=rng)
    elif args.world == "osm":
        from chainsolvers_eval.osm import hannover_osm_world
        world = hannover_osm_world(args.persons, distance_noise=args.noise, rng=rng)
    elif args.world == "two-zone":
        world = two_zone_world(n_persons=args.persons, distance_noise=args.noise,
                               heavy_tail=args.heavy_tail, rng=rng)
    else:
        world = generate_world(n_locations=args.locations, n_persons=args.persons,
                               gravity_scale=args.gravity_scale, distance_noise=args.noise, rng=rng)
    path = plot_world(world, args.out, n_chains=args.n_chains, seed=args.seed)
    m = world.meta
    print(f"Wrote {path}  ({m['n_persons']} persons, {m['n_legs']} legs, "
          f"{m['n_free_legs']} to place, {len(world.topology.types)} types)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

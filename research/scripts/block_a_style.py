"""Canonical plot style for ALL chainsolvers paper figures (Block A and Block B).

House style: every solver gets a fixed (colour, marker, linestyle) triple, drawn with HOLLOW marker
faces (mfc="none") so overlapping curves stay readable. Use `line(ax, x, y, solver)` for every line
plot, and `COL[solver]` for bar/patch colours. Keeping this in one module means Block A, Block B, the
robust companions, and any future figure share identical styling -- import from here, do not redefine.
"""

# (colour, marker, linestyle) per solver -- consistent across EVERY figure/panel.
STYLE = {
    "carla":           ("#1f77b4", "^", ":"),
    "dp_rings":        ("#ff7f0e", "o", "--"),
    "dp_carla":        ("#2ca02c", "s", "--"),
    "dp_rings_refine": ("#d62728", "v", "-."),
    "dp_carla_refine": ("#9467bd", "D", "-."),
    "dp_carla_pot":    ("#1a9850", "8", "-"),   # potential-pooled near-oracle (Block B)
    "dp_full":         ("#8c564b", "*", "-"),
    "rda":             ("#e377c2", "X", "--"),
    "rda_guided":      ("#7f7f7f", "P", "-."),
    "dp_sample":       ("#bcbd22", "p", "-"),
    "dp_sample_capped": ("#5ab4ac", "<", "-"),  # dp_sample + candidate cap (fast variant)
    "carla_sample":    ("#e6ab02", "d", "-"),   # CARLA greedy-ancestral sampler (ablation vs dp_sample)
    "gravity_independent": ("#a6761d", ">", "--"),  # no-coupling gravity floor (alpha-independent)
    "dp_sample_tuned": ("#17becf", "h", "-"),
}
COL = {k: v[0] for k, v in STYLE.items()}  # colour lookup (bars/patches)
_DEFAULT_STYLE = ("#333333", "o", "-")


def line(ax, x, y, solver, *, ms=5, yerr=None, capsize=2, label=None, **kw):
    """House-style line for `solver`: per-solver colour+marker+linestyle, hollow faces.
    Pass yerr for an errorbar; label defaults to the solver name."""
    c, mk, ls = STYLE.get(solver, _DEFAULT_STYLE)
    common = dict(color=c, marker=mk, ls=ls, ms=ms, lw=1.5, mfc="none", mew=1.3,
                  label=solver if label is None else label)
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, capsize=capsize, **common, **kw)
    else:
        ax.plot(x, y, **common, **kw)

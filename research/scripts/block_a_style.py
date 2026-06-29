"""Canonical plot style for ALL chainsolvers paper figures (Block A and Block B).

House style: every solver gets a fixed (colour, marker, linestyle) triple, drawn with HOLLOW marker
faces (mfc="none") so overlapping curves stay readable. Use `line(ax, x, y, solver)` for every line
plot, and `COL[solver]` for bar/patch colours. Keeping this in one module means Block A, Block B, the
robust companions, and any future figure share identical styling -- import from here, do not redefine.
"""

# (colour, marker, linestyle) per solver -- consistent across EVERY figure/panel.
STYLE = {
    # Solvers are grouped into HUE FAMILIES by role; members share a hue but are spaced apart
    # (light->dark) so all 14 stay separable in one plot while still reading as the same family.
    # --- CARLA family (heuristic placement) -- blues ---
    "carla":           ("#3182bd", "^", ":"),   # darker than the old sky-blue so the dotted line reads clearly
    "carla_sample":    ("#08519c", "d", "-"),   # CARLA greedy-ancestral sampler (ablation vs dp_sample); deeper blue, still distinct from carla
    # --- DP argmin family (exact / pruned) -- greens, light->dark by elaboration ---
    "dp_rings":        ("#b8e3b2", "o", "--"),
    "dp_carla":        ("#80ca80", "s", "--"),
    "dp_rings_refine": ("#3fa85b", "v", "-."),
    "dp_carla_refine": ("#147e3a", "D", "-."),
    "dp_carla_pot":    ("#004d1f", "8", "-"),   # potential-pooled near-oracle (Block B)
    # --- Oracle -- black ---
    "dp_full":         ("#111111", "*", "-"),
    # --- Generative samplers -- oranges, light->dark ---
    "dp_sample":       ("#fd9a4e", "p", "-"),
    "dp_sample_capped": ("#cc4c02", "<", "-"),  # dp_sample + candidate cap (fast variant)
    # --- RDA baselines -- magenta ---
    "rda":             ("#e7298a", "X", "--"),
    "rda_guided":      ("#ae017e", "P", "-."),
    # --- Naive floor -- neutral gray ---
    "gravity_independent": ("#808080", ">", "--"),  # no-coupling gravity floor (alpha-independent)
}
COL = {k: v[0] for k, v in STYLE.items()}  # colour lookup (bars/patches)
_DEFAULT_STYLE = ("#333333", "o", "-")

# Canonical legend/display label per solver -- MUST match the paper's Table 1 / prose names.
# The DP family is named by candidate generation: DP (ring envelopes) -> DP-circle (+circle, n=2),
# with suffixes -R (refinement), -pot (potential pool), -full (oracle). `line()` uses this by default,
# so legends read like the paper; pass label= to override. LABEL.get(key, key) is the safe lookup.
LABEL = {
    "carla":               "CARLA",
    "carla_sample":        "CARLA-sample",
    "dp_rings":            "DP",
    "dp_carla":            "DP-circle",
    "dp_rings_refine":     "DP-R",
    "dp_carla_refine":     "DP-circle-R",
    "dp_carla_pot":        "DP-pot",
    "dp_full":             "DP-full",
    "dp_sample":           "DP-sample",
    "dp_sample_capped":    "DP-sample (capped)",
    "rda":                 "RDA",
    "rda_guided":          "RDA-guided",
    "gravity_independent": "GravityInd",
}

# Canonical display names for the worlds -- use EVERYWHERE a world appears in a title/label so the
# raw keys ("two_zone") never leak. WORLD_NAME.get(key, key) is the safe lookup.
WORLD_NAME = {
    "gauss_hannover": "Gauss-Hannover",
    "osm_hannover": "OSM-Hannover",
    "two_zone": "Two-zone",
}

# Canonical typography for EVERY paper figure -- one scheme so all graphs match. Call
# apply_paper_style() once at the top of each figure script (before plotting); per-call
# fontsize= overrides are then unnecessary and should be removed so this governs.
# Headers (per-plot title + suptitle) are deliberately the largest -- the world-name panel
# headers (e.g. "Gauss-Hannover", "Two-zone") read clearly; sized so the longest still fits.
FS_LEGEND, FS_LABEL, FS_TICK, FS_TITLE, FS_SUPTITLE = 11, 11, 9, 14, 14


def apply_paper_style(whitegrid=True):
    """Canonical font sizes (legend 9 / labels 11 / ticks 9 / title 12 / suptitle 13) for every
    figure. With whitegrid=True (default) also applies the seaborn whitegrid theme; pass
    whitegrid=False to keep plain matplotlib styling (the 'old' look) while still getting the
    consistent sizes -- used by the vs-MiD distribution figures. Solver colours stay explicit via
    STYLE, so the seaborn palette is irrelevant. Idempotent."""
    import matplotlib as mpl
    if whitegrid:
        import seaborn as sns
        sns.set_theme(style="whitegrid")
    mpl.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white", "savefig.bbox": "tight",
        "axes.titlesize": FS_TITLE, "axes.titleweight": "regular",
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND, "legend.title_fontsize": FS_LEGEND, "legend.loc": "best",
        "figure.titlesize": FS_SUPTITLE,
    })


def line(ax, x, y, solver, *, ms=5, yerr=None, capsize=2, label=None, **kw):
    """House-style line for `solver`: per-solver colour+marker+linestyle, hollow faces.
    Pass yerr for an errorbar; label defaults to the canonical paper name (LABEL)."""
    c, mk, ls = STYLE.get(solver, _DEFAULT_STYLE)
    common = dict(color=c, marker=mk, ls=ls, ms=ms, lw=1.5, mfc="none", mew=1.3,
                  label=LABEL.get(solver, solver) if label is None else label)
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, capsize=capsize, **common, **kw)
    else:
        ax.plot(x, y, **common, **kw)


# Canonical legend order = the STYLE key order: family-grouped (CARLA blues, DP greens, oracle,
# samplers, RDA, floor) with the DP block in the paper's Table-1 order (DP, DP-circle, DP-R,
# DP-circle-R, DP-pot). legend() reorders entries to this REGARDLESS of draw order.
_LEGEND_RANK = {k: i for i, k in enumerate(STYLE)}


def legend(ax, **kw):
    """ax.legend() but entries reordered by canonical family order (STYLE key order). Labels are
    matched to a solver via LABEL -- exact first, then the longest display-name prefix (so composite
    labels like 'DP-sample: informed' or 'DP-circle-R argmin' still sort into their family).
    Unmatched labels (e.g. 'true (DGP)') keep their original order at the end. Pass any ax.legend kw."""
    handles, labels = ax.get_legend_handles_labels()
    inv = {disp: key for key, disp in LABEL.items()}
    disp_by_len = sorted(inv, key=len, reverse=True)

    def rank(lbl, i):
        if lbl in inv:
            return (_LEGEND_RANK.get(inv[lbl], 10**6), i)
        for disp in disp_by_len:
            if lbl.startswith(disp):
                return (_LEGEND_RANK.get(inv[disp], 10**6), i)
        return (10**6 + 1, i)

    idx = sorted(range(len(labels)), key=lambda i: rank(labels[i], i))
    return ax.legend([handles[i] for i in idx], [labels[i] for i in idx], **kw)

"""Calibration-form (structural transform) ablation for Block C, gauss-Hannover, fixed anchors.

The generative samplers calibrate their attractiveness sensitivity (alpha) by MLE under a chosen
attractiveness transform (`log` / `log1p` / `linear`). `log` matches the gravity DGP. This figure
shows the consequence of mis-specifying that form on the forecast district-share curve:
the matched `log` tracks the true counterfactual; `log1p` over-shoots, `linear` under-shoots
(it flattens the elasticity). The argmin reference (dp_carla, w1) is form-robust by construction.

Reads the transform-tagged CSVs produced by block_c.py (run thrice with --transform / --no-plot,
preserved as ..._<world>__<transform>.csv). Pure post-hoc; data is final.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from block_a_style import apply_paper_style, WORLD_NAME

ROOT = os.path.join(os.path.dirname(__file__), "..")
INDIR = os.path.join(ROOT, "out", "block_c_calibform")
WORLD = "gauss_hannover"
TRANSFORMS = ["log", "log1p", "linear"]
TFCOL = {"log": "#2ca02c", "log1p": "#d62728", "linear": "#1f77b4"}
TFLAB = {"log": "log (matched DGP)", "log1p": "log1p", "linear": "linear"}
GEN = [("dp_sample", "dp_sample (forward-backward)"),
       ("carla_sample", "carla_sample (greedy-ancestral)")]


def _load(t):
    return pd.read_csv(os.path.join(INDIR, f"prognosis_attractiveness_fixed_{WORLD}__{t}.csv"))


def _curve(df, solver, condition):
    s = df[(df.solver == solver) & (df.condition == condition)].sort_values("level")
    return s["level"].to_numpy(float), s["outcome_main"].to_numpy(float) * 100.0


def main():
    apply_paper_style()
    data = {t: _load(t) for t in TRANSFORMS}
    any_df = data["log"]
    fig, axes = plt.subplots(1, len(GEN), figsize=(11, 4.6), sharex=True, sharey=True)

    for ax, (solver, title) in zip(axes, GEN):
        # true counterfactual -- identical across transforms (GT independent of calibration)
        xt, yt = _curve(any_df, "true", "true")
        ax.plot(xt, yt, color="black", marker="o", ls="-", lw=2.0, ms=6, mfc="white",
                mew=1.4, label="true counterfactual", zorder=6)
        # generative forecast per transform
        for t in TRANSFORMS:
            a = data[t].alpha_cal.iloc[0]
            x, y = _curve(data[t], solver, "informed")
            ax.plot(x, y, color=TFCOL[t], marker="s", ls="--", lw=1.6, ms=5, mfc="none",
                    mew=1.3, label=f"{TFLAB[t]}  ($\\hat\\alpha$={a:.2f})")
        ax.set(title=title, xlabel="attractiveness boost $b$")
        ax.set_xticks([1, 2, 4, 8])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("district share (%)")
    axes[0].legend(frameon=True, fontsize=8)
    fig.suptitle(f"Calibration-form ablation -- {WORLD_NAME.get(WORLD, WORLD)}, fixed anchors: "
                 f"forecast vs structural transform")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        out = os.path.join(INDIR, f"C_calibform_ablation_{WORLD}.{ext}")
        fig.savefig(out, dpi=130)
        print("wrote", out)


if __name__ == "__main__":
    main()

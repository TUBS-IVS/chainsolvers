"""Palette swatch: render every solver's (colour, marker, linestyle) so the house style can be
eyeballed for clashes. Reads STYLE from block_a_style -- single source of truth. Output: out/palette_swatch.png."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from block_a_style import STYLE, apply_paper_style, line

apply_paper_style(whitegrid=False)

solvers = list(STYLE)
n = len(solvers)
fig, (axc, axl) = plt.subplots(1, 2, figsize=(11, 0.42 * n + 1))

# Left: flat colour chips with the hex + name.
for i, s in enumerate(solvers):
    c, mk, ls = STYLE[s]
    y = n - 1 - i
    axc.add_patch(plt.Rectangle((0, y - 0.4), 1, 0.8, color=c))
    axc.text(1.1, y, f"{s}", va="center", ha="left", fontsize=10)
    axc.text(3.9, y, c, va="center", ha="right", fontsize=9, family="monospace", color="#555")
axc.set_xlim(0, 4); axc.set_ylim(-0.6, n - 0.4); axc.axis("off")
axc.set_title("colour chips")

# Right: sample line so marker + linestyle (the in-plot disambiguators) are visible.
x = np.linspace(0, 1, 6)
for i, s in enumerate(solvers):
    y = n - 1 - i
    line(axl, x, np.full_like(x, y), s, label=None)
    axl.text(1.05, y, s, va="center", ha="left", fontsize=10)
axl.set_xlim(0, 2.2); axl.set_ylim(-0.6, n - 0.4); axl.axis("off")
axl.set_title("colour + marker + linestyle (as drawn)")

out = Path(__file__).resolve().parents[1] / "out" / "palette_swatch.png"
out.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"wrote {out}")

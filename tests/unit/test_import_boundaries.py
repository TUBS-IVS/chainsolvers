"""Enforce the library/research boundary.

The library (``src/chainsolvers``) must never import the research package
(``chainsolvers_eval``). Research may import the library, never the reverse.
This keeps the distributed wheel free of research code and dependencies.
"""

from pathlib import Path

LIBRARY_ROOT = Path(__file__).resolve().parents[2] / "src" / "chainsolvers"


def test_library_never_imports_research():
    offenders = []
    for py in LIBRARY_ROOT.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            if "chainsolvers_eval" in line:
                offenders.append(f"{py.relative_to(LIBRARY_ROOT.parent)}:{lineno}: {stripped}")

    assert not offenders, (
        "Library code must not import the research package (chainsolvers_eval):\n"
        + "\n".join(offenders)
    )

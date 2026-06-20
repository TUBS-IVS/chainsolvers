"""Research / evaluation scaffolding for chainsolvers.

NOT part of the library's public API and NOT imported by ``chainsolvers/__init__.py`` —
the core (`setup`, `solve`, solvers, pipeline) never depends on anything here. This
subpackage holds the synthetic-world generator, model calibration, and the survey-style
evaluation primitives used for benchmarking and the paper experiments (see `scripts/`).

Importing it is opt-in, e.g. ``from chainsolvers_eval import synth``.
"""

# Contributing to Lumenairy

Thanks for your interest!  This document covers how to get a working
development environment, run the validation suite, and submit a pull
request.

## Quick start

```bash
git clone https://github.com/travaj24/Lumenairy.git
cd Lumenairy
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all,dev,gui]"
```

The `[all]` extra pulls in `pyfftw`, `numexpr`, `numba`, `astropy`,
`h5py`, `zarr`, `psutil` (everything optional for the core library).
`[gui]` adds the PySide6 + pyvista stack for the Optical Designer
GUI.  `[dev]` adds `pytest` for the test runners.

## Running the validation suite

The library has a topic-organized validation suite at
[`validation/`](./validation/).  Each `test_*.py` is self-contained
and can be run individually, or all together via the runner:

```bash
python validation/run_all.py            # full suite (~2 min)
python validation/run_all.py --quiet    # only per-file pass/fail + summary
python validation/run_all.py test_lenses test_propagation
```

A passing run ends with `ALL N files passed.`  Each file uses the
in-house `Harness` (see `validation/_harness.py`); harness asserts
do **not** abort on first failure -- they collect results and print
a summary at the end so you can see the full failure surface.

There are currently 16 files / ~298 assertions.  CI runs the full
suite on every push and PR via
[`.github/workflows/validate.yml`](./.github/workflows/validate.yml).

## Code style

Pure Python, no formatter enforced.  Match the surrounding code:

- 4-space indents
- ~80-character soft line limit (we don't break long-string asserts
  for it, but everything else)
- `from foo import bar` over `import foo` when only one or two names
  are used
- Top-of-module docstring describing what the module does, scientific
  conventions used (sign of `exp(-i*omega*t)`, units in SI metres,
  etc.), and authorship

The library is heavy on **comments that explain *why* code is shaped
the way it is** -- particularly around numerical-stability mitigations,
performance trade-offs, and "we tried the obvious thing and it broke
because X".  That's intentional -- it pays for itself the first time
someone (us, in three months) tries to "simplify" a load-bearing
piece.  Lean toward keeping these comments.

Avoid:

- adding new public API without docstrings
- silently changing numerical defaults without a CHANGELOG entry
- adding imports inside hot loops (move them to module top)

## Submitting a pull request

1. Fork + branch from `main`.
2. Make your change.  Keep the diff focused -- one PR per logical
   change.  Refactors that mix with feature work are hard to review.
3. Run `python validation/run_all.py` locally.  All 16 files must
   pass.  If your change adds new behaviour, add at least one test
   in the appropriate topic file.
4. Add a CHANGELOG entry under `## [Unreleased]` (we'll cut the
   version on merge).
5. Open the PR with a description that covers:
   - **What** changed (1-2 sentences)
   - **Why** -- the problem you're solving or feature you're adding
   - **How** -- algorithmic / numerical notes if relevant
   - **Testing** -- which validation files cover the change

Tag `@travaj24` for review.  CI must be green before merge.

## Reporting issues

Bugs / questions / feature requests: open an issue at
<https://github.com/travaj24/Lumenairy/issues>.

For physics or numerical correctness questions, please include:

- a minimal reproducer (the smallest prescription / grid / call
  sequence that exhibits the issue)
- the expected and actual numerical result
- your platform (`python --version`, `numpy.__version__`, OS)

For performance reports, include grid size (`N`, `dx`), wavelength,
which functions are hot in your profile, and which optional backends
are active (`PYFFTW_AVAILABLE`, `NUMEXPR_AVAILABLE`, etc.).

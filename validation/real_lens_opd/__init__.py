"""Real-lens OPD validation.

Compares ``apply_real_lens`` wave OPD to geometric ray-traced OPL for a
suite of reference lenses.  Both ``slant_correction=True`` (new default)
and ``slant_correction=False`` (old paraxial) are tested, producing
side-by-side plots and a summary report.

Usage
-----
    python -m validation.real_lens_opd.run_validation

Outputs land in ``validation/real_lens_opd/results/``.
"""

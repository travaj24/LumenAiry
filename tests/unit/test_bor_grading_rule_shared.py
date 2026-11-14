"""The body-of-revolution solvers use the ONE element-grading rule of the
spectral-element family (``pmm._core._graded_fractions``) instead of two
inline copies of the Chebyshev-Lobatto formula, bit for bit (2026-11-14)."""
import ast
import pathlib

import numpy as np
import pytest

import lumenairy.elements.bor.bor_stack as BS
import lumenairy.elements.bor._jax_sem as JSEM
from lumenairy.elements.pmm._core import _graded_fractions


def _frozen_inline(k, grade):
    """The formula both BOR modules carried inline before (verbatim)."""
    if grade:
        return 0.5 * (1.0 - np.cos(np.pi * np.arange(k + 1) / k))
    return np.linspace(0.0, 1.0, k + 1)


@pytest.mark.parametrize("grade", [True, False])
def test_shared_rule_is_bit_identical_to_the_former_inline_copies(grade):
    for k in range(2, 41):
        assert np.array_equal(_graded_fractions(k, grade),
                              _frozen_inline(k, grade)), k


def test_no_inline_grading_formula_left_in_bor():
    for mod in (BS, JSEM):
        src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
        assert "np.cos(np.pi * np.arange(k + 1) / k)" not in src, mod.__name__
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "_graded_fractions" in names, mod.__name__


def test_bor_meshes_unchanged():
    """The traced mesh builder of the JAX BOR path, on concrete input, with
    graded sub-elements: equal to the frozen inline recipe."""
    class _St:
        Rbig = 3e-6
        elements_per_segment = 4
        grade = True
        degree = 8
        n_mesh_cap = None
    segs = [(1e-6, (2.25, 2.25, 2.25)), (2e-6, (1.0, 1.0, 1.0)),
            (3e-6, (1.0, 1.0, 1.0))]
    bnd, _eps = JSEM._build_bnd(_St, [1e-6, 2e-6], segs, 2 * np.pi / 1e-6)
    assert bnd is not None
    raw = np.array([0.0, 1e-6, 2e-6, 3e-6])
    sloc = _frozen_inline(4, True)
    exp = [raw[0]]
    for lo, hi in zip(raw[:-1], raw[1:]):
        exp.extend(lo + (hi - lo) * sloc[1:])
    exp = np.asarray(exp)
    # the wavelength-resolution cap may split further; every graded boundary
    # of the frozen recipe must be present bit for bit
    for x in exp:
        assert np.any(np.asarray(bnd) == x), x

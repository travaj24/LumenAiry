"""V7 -- the OTHER copies of the same branch test.

The audit's blast-radius section says ``_sqrt_decay`` "is shared:
``rcwa/oned.py``, ``rcwa/twod.py``, ``rcwa/stack.py``, ``pmm/twod.py``,
``pmm/_jax_twod.py``, ``pmm/_jax_stack2d.py``, ``pmm/_jax_twod_jones.py`` and
``elements/berreman.py`` all call it, and every one now gets the pinned root."

Grep says otherwise for the PMM half of that list.  ``pmm/twod.py:411``,
``pmm/twod_staggered.py:2084``, ``pmm/_jax_stack2d.py:175``,
``pmm/_jax_twod.py:341`` and ``pmm/_jax_twod_jones.py:191`` each define their
OWN ``_sqrt_decay``, textually identical to the pre-fix RCWA one, exact
``r.real == 0`` pin included; none of them imports the patched function.  So
this probe asks the physical question rather than the textual one: do those
copies carry the same defect on a coincidence fixture, and does installing the
fixed body in them repair it?

Three arms per fixture, in one process:
  as_shipped   the tree as it stands
  pmm_fixed    the PMM copies replaced by the FIXED body (band + conj)
  rcwa_pre     the RCWA copy reverted to the pre-fix body (control: shows the
               fixture is a coincidence fixture at all where RCWA is involved)

Usage: python v7_pmm_copies.py <out.json>
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402

P, WL, DEPTH = V._P, V._WL, V._DEPTH

_PMM_MODULES = ("lumenairy.elements.pmm.twod",
                "lumenairy.elements.pmm.twod_staggered")


def _fixed_body(x):
    """The SHIPPED rcwa ``_sqrt_decay``, re-expressed for a NumPy-only copy."""
    x = np.asarray(x, dtype=complex)
    r = np.sqrt(x)
    scale = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
    on_cut = np.abs(r.real) <= 1e-8 * scale
    return np.where(on_cut & (r.imag < 0), np.conj(r), r)


class PmmFixed:
    """Install the fixed body in every PMM module that carries its own copy."""

    def __init__(self):
        self._saved = []

    def __enter__(self):
        for name in _PMM_MODULES:
            mod = importlib.import_module(name)
            self._saved.append((mod, mod._sqrt_decay))
            mod._sqrt_decay = _fixed_body
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


def source_audit():
    """Which modules bind which ``_sqrt_decay``, read off the live objects."""
    rows = {}
    import inspect
    for name in ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
                 "lumenairy.elements.rcwa.stack", "lumenairy.elements.berreman",
                 "lumenairy.elements.pmm.twod",
                 "lumenairy.elements.pmm.twod_staggered"):
        try:
            mod = importlib.import_module(name)
        except Exception as exc:                              # pragma: no cover
            rows[name] = dict(error=repr(exc))
            continue
        fn = getattr(mod, "_sqrt_decay", None)
        if fn is None:
            rows[name] = dict(defines=False)
            continue
        try:
            src = inspect.getsource(fn)
        except Exception:                                     # pragma: no cover
            src = ""
        rows[name] = dict(
            defines=True,
            defined_in=getattr(fn, "__module__", None),
            has_exact_zero_pin=("r.real == 0" in src),
            has_relative_band=("_CUT_BAND_REL" in src or "1e-8 * scale" in src),
            flips_with_conj=("conj" in src))
    # the JAX / closure copies are nested functions: read the FILE instead
    import lumenairy
    root = os.path.dirname(lumenairy.__file__)
    for rel in ("elements/pmm/_jax_twod.py", "elements/pmm/_jax_stack2d.py",
                "elements/pmm/_jax_twod_jones.py"):
        txt = open(os.path.join(root, rel), encoding="utf-8",
                   errors="replace").read()
        rows[rel] = dict(defines=("def _sqrt_decay" in txt),
                         has_exact_zero_pin=("on_cut = r.real == 0" in txt),
                         has_relative_band=("_CUT_BAND_REL" in txt))
    return rows


# -------------------------------------------------------------------- probes
# The entry points that reach ``pmm/twod.py``'s OWN ``_sqrt_decay`` with a
# LAYER eigenvalue (``_layer_modes_projected`` line 669, ``_symmetric_solve_2d``
# line 756) rather than only with a homogeneous REGION's exact ``-kz^2``
# (``_homogeneous_modes`` line 779).  ``pmm_jones_2d`` is NOT one of them: its
# layer goes through ``rcwa._core._layer_eigenmodes_tensor``, i.e. through the
# FIXED function -- which is why it looks clean.
_REC = []


class PmmSpy:
    """Record every array handed to each PMM copy of ``_sqrt_decay``."""

    def __init__(self):
        self._saved = []

    def __enter__(self):
        _REC.clear()
        for name in _PMM_MODULES:
            mod = importlib.import_module(name)
            orig = mod._sqrt_decay
            self._saved.append((mod, orig))

            def make(orig=orig, tag=name):
                def spy(x):
                    a = np.asarray(x, dtype=complex)
                    _REC.append((tag, a.copy()))
                    return orig(x)
                return spy
            mod._sqrt_decay = make()
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


def _census_recorded():
    """For every array the PMM copies were handed: how many modes are
    numerically ON the cut, how many of those have an EXACT zero imaginary part
    (where the old pin does fire) and how many carry a rounding-level imaginary
    part with the INCOMING sign (where it does not, and the root is decided by
    the eigensolver's last bit)."""
    tot = on_cut = exact_zero_im = noisy_im = incoming_after = 0
    worst_im = 0.0
    tags = set()
    for tag, a in _REC:
        tags.add(tag)
        r = np.sqrt(a)
        scale = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
        oc = np.abs(r.real) <= 1e-8 * scale
        tot += int(a.size)
        on_cut += int(oc.sum())
        if not oc.any():
            continue
        im = a.imag[oc]
        exact_zero_im += int(np.sum(im == 0.0))
        noisy_im += int(np.sum(im != 0.0))
        worst_im = max(worst_im, float(np.max(np.abs(im))))
        # what the SHIPPED (old) pin leaves behind:
        old = np.where((r.real == 0) & (r.imag < 0), -r, r)
        incoming_after += int(np.sum(oc & (old.imag < 0)))
    return dict(tags=sorted(tags), modes=tot, on_cut=on_cut,
                on_cut_exact_zero_imag=exact_zero_im,
                on_cut_rounding_imag=noisy_im,
                max_abs_imag_on_cut=worst_im,
                incoming_after_old_pin=incoming_after)


def _pmm_jones_2d(n_sub, twist=0.7, bg=2.25):
    from lumenairy.elements.pmm import pmm_jones_2d
    return pmm_jones_2d(P, P, V.uniaxial_cell(S=32, twist=twist, bg=bg), n_sub,
                        1.0, DEPTH, WL, degree=5, n_orders=5)


def _pmm_eff_2d(n_sub, eps_pillar=4.0, eps_host=2.25):
    from lumenairy.elements.pmm import pmm_efficiency_2d
    return pmm_efficiency_2d(P, P, eps_pillar, eps_host, (0.25, 0.75),
                             (0.25, 0.75), n_sub, 1.0, DEPTH, WL, degree=5,
                             n_orders=5)


def _pmm_stag(n_sub, blk=4.0, bg=2.25):
    from lumenairy.elements.pmm import pmm_jones_2d_staggered
    return pmm_jones_2d_staggered(P, P, V.scalar_cell(S=16, bg=bg, blk=blk),
                                  n_sub, 1.0, DEPTH, WL, degree=4, n_orders=3)


def _pmm_cell(n_sub, blk=4.0, bg=2.25, theta=0.0, phi=0.0, sym="auto"):
    from lumenairy.elements.pmm import pmm_efficiency_2d_cell
    return pmm_efficiency_2d_cell(P, P, V.scalar_cell(S=32, bg=bg, blk=blk),
                                  n_sub, 1.0, DEPTH, WL, degree=5, n_orders=5,
                                  theta=theta, phi=phi, symmetry=sym)


def _pmm_hybrid(n_sub, blk=4.0, bg=2.25):
    from lumenairy.elements.pmm import PMM2DStackHybrid
    st = PMM2DStackHybrid(P, P, n_superstrate=1.0, n_substrate=n_sub,
                          degree=5, n_orders=5)
    st.add_layer(DEPTH, eps_cell=V.scalar_cell(S=32, bg=bg, blk=blk))
    return st.set_source(WL).solve()


FIXTURES = {
    "pmm_cell_coinc": (lambda: _pmm_cell(1.5), "eff", True),
    "pmm_cell_weakmod_coinc": (lambda: _pmm_cell(1.5, blk=2.25 + 1e-6),
                               "eff", True),
    "pmm_cell_offcoinc": (lambda: _pmm_cell(1.63), "eff", False),
    "pmm_cell_full_coinc": (lambda: _pmm_cell(1.5, sym=False), "eff", True),
    "pmm_cell_oblique_coinc": (lambda: _pmm_cell(1.5, theta=0.3), "eff", True),
    "pmm_cell_conical_coinc": (lambda: _pmm_cell(1.5, theta=0.2, phi=0.7),
                               "eff", True),
    # coincidence: the layer background 2.25 IS n_substrate^2
    "pmm_jones_2d_coinc": (lambda: _pmm_jones_2d(1.5), "jones", True),
    "pmm_jones_2d_offcoinc": (lambda: _pmm_jones_2d(1.63), "jones", False),
    "pmm_jones_2d_weakmod_coinc":
        (lambda: _pmm_jones_2d(1.5, twist=0.0, bg=2.25), "jones", True),
    "pmm_eff_2d_coinc": (lambda: _pmm_eff_2d(1.5), "eff", True),
    "pmm_eff_2d_weakmod_coinc":
        (lambda: _pmm_eff_2d(1.5, eps_pillar=2.25 + 1e-6), "eff", True),
    "pmm_eff_2d_offcoinc": (lambda: _pmm_eff_2d(1.63), "eff", False),
    "pmm_stag_coinc": (lambda: _pmm_stag(1.5), "jones", True),
    "pmm_stag_weakmod_coinc":
        (lambda: _pmm_stag(1.5, blk=2.25 + 1e-6), "jones", True),
    "pmm_stag_offcoinc": (lambda: _pmm_stag(1.63), "jones", False),
}


def _closure(res, kind):
    return (V.closure_defect_jones(res) if kind == "jones"
            else V.closure_defect_eff(res))


def _arrays(res):
    out = []
    for x in list(res):
        a = np.asarray(x)
        if a.dtype.kind in "fc":
            out.append(a)
    return out


def run(fn, kind):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = fn()
            return dict(closure=_closure(res, kind),
                        arrays=_arrays(res),
                        warnings=sorted({w.category.__name__ for w in caught}))
        except Exception as exc:
            return dict(error=repr(exc)[:150], arrays=None,
                        closure=float("nan"), warnings=[])


def main():
    V.require_local_tree()
    out = sys.argv[1]
    V.claim_output(out)
    audit = source_audit()
    print("WHICH MODULES CARRY THE OLD EXACT-ZERO PIN")
    for k, v in audit.items():
        print("  %-42s %s" % (k, v))

    rows = {}
    for name, (fn, kind, coinc) in FIXTURES.items():
        with PmmSpy():
            shipped = run(fn, kind)
            cens = _census_recorded()
        with PmmFixed():
            fixed = run(fn, kind)
        move = None
        if shipped["arrays"] is not None and fixed["arrays"] is not None:
            move = max(
                (float(np.max(np.abs(a - b))) if a.shape == b.shape else
                 float("inf"))
                for a, b in zip(shipped["arrays"], fixed["arrays"]))
        rows[name] = dict(
            coincident=coinc,
            closure_as_shipped=shipped["closure"],
            closure_pmm_fixed=fixed["closure"],
            warnings_as_shipped=shipped["warnings"],
            warnings_pmm_fixed=fixed["warnings"],
            error_as_shipped=shipped.get("error"),
            error_pmm_fixed=fixed.get("error"),
            max_move_when_pmm_copies_are_fixed=move,
            bit_identical=(move == 0.0) if move is not None else None,
            pmm_copy_census=cens)
        r = rows[name]
        c = r["pmm_copy_census"]
        print("%-26s coinc=%-5s closure shipped=%+.3e pmm-fixed=%+.3e move=%-11s"
              " | PMM copy: modes=%-5s onCut=%-4s exact0Im=%-4s noisyIm=%-4s "
              "maxImOnCut=%.2e incomingAfterOldPin=%s"
              % (name, coinc, r["closure_as_shipped"], r["closure_pmm_fixed"],
                 ("%.3e" % move) if move is not None else "-",
                 c["modes"], c["on_cut"], c["on_cut_exact_zero_imag"],
                 c["on_cut_rounding_imag"], c["max_abs_imag_on_cut"],
                 c["incoming_after_old_pin"]))
    V.dump(out, dict(source_audit=audit, fixtures=rows))


if __name__ == "__main__":
    main()

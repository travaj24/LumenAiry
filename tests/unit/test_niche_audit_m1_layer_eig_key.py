"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 finding M1 (HIGH, silent physics
error): the ``RCWAStack`` shapes-layer eigenmode cache key collided.

``RCWAStack._layer_eig_key`` dedupes the (expensive) per-layer eigenproblem
inside one solve.  For a ``shapes`` layer it flattened EVERY shape's
``(key, repr(value))`` pairs into ONE sorted multiset::

    tuple(sorted((k, repr(v)) for s in shapes for k, v in s.items()))

so two structurally DIFFERENT shape lists carrying the same multiset of pairs
produced the SAME key and the second layer silently reused the first layer's
eigenmodes.  The canonical collision is a pair of shapes with their CENTRES
exchanged:

* ``A = [disk(r=0.20 um) @ (0.25P, 0.25P), disk(r=0.10 um) @ (0.70P, 0.60P)]``
* ``B =`` the same two disks with the centres swapped.

Measured on this file's 2-layer stack before the fix (n_orders = 3):

* ``_layer_eig_key(A) == _layer_eig_key(B)`` -> True;
* the ``(A, B)`` stack returned BIT-IDENTICALLY the ``(A, A)`` answer
  (``max|dR| = max|dT| = 0``);
* against a rasterised ``eps_cell`` oracle of the SAME (A, B) geometry:
  zeroth-order ``R0 = 0.012394`` vs the oracle's ``0.002617`` (3.7x relative
  error), ``max|dR| = 2.2e-2``, ``max|dT| = 2.0e-1``;
* energy closure ``|R + T - 1| = 5.8e-15`` throughout -- the wrong answer was
  perfectly energy-conserving, so no tripwire could see it.

After the fix (per-shape, order-preserving key) the same probes give: keys
differ, ``(A, B)`` differs from ``(A, A)`` by ``max|dR| = 2.2e-2``, and
``(A, B)`` agrees with the raster oracle to ``max|dR| = 3.7e-5`` /
``max|dT| = 1.9e-4`` (the honest analytic-form-factor-vs-raster level: the
never-colliding ``(A, A)`` control agrees to ``1.9e-5`` / ``6.6e-5`` at the
same raster).  The tolerances below carry >= 10x headroom over the measured
agreement and >= 20x margin below the bug's error.

The geometry is deliberately fold-INELIGIBLE (the two disks are not
centro-symmetric about any common centre), so the default
``symmetry='auto'`` even-parity fast path bails and the DEFAULT user path
exercises the mode cache.  (A first attempt used disks at (0.25P, 0.25P) and
(0.75P, 0.75P); that pair IS centro-symmetric about (0.25P, 0.25P), the fold
engaged, and the fold -- which builds every layer's spec independently -- hid
the collision.)
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa.stack import RCWAStack

_WL = 0.55e-6
_PX = _PY = 0.9e-6
_NORD = 3                      # 49 retained orders -> ~0.1 s per stack solve
_TH = 0.20e-6
_EPS_S = 6.0
_EPS_BG = 1.0
_C1 = (0.25 * _PX, 0.25 * _PY)
_C2 = (0.70 * _PX, 0.60 * _PY)
_R_BIG, _R_SML = 0.20e-6, 0.10e-6

SH_A = [{"shape": "disk", "radius": _R_BIG, "center": _C1, "eps": _EPS_S},
        {"shape": "disk", "radius": _R_SML, "center": _C2, "eps": _EPS_S}]
SH_B = [{"shape": "disk", "radius": _R_BIG, "center": _C2, "eps": _EPS_S},
        {"shape": "disk", "radius": _R_SML, "center": _C1, "eps": _EPS_S}]

_RASTER_S = 481                # oracle sampling (measured convergence below)


def _stack():
    return RCWAStack(_PX, period_y=_PY, n_superstrate=1.0, n_substrate=1.5,
                     n_orders=_NORD, n_orders_y=_NORD)


def _shapes_stack(shape_lists, theta=0.0, phi=0.0):
    st = _stack()
    for sh in shape_lists:
        st.add_layer(_TH, shapes=sh, eps_background=_EPS_BG)
    return st.set_source(_WL, theta=theta, phi=phi)


def _raster(shapes, S=_RASTER_S):
    """The same geometry as a pixel cell (pixel-centre sampling)."""
    x = (np.arange(S) + 0.5) * _PX / S
    y = (np.arange(S) + 0.5) * _PY / S
    xx, yy = np.meshgrid(x, y, indexing="ij")
    cell = np.full((S, S), _EPS_BG, dtype=complex)
    for sh in shapes:
        cx, cy = sh["center"]
        r = sh["radius"]
        cell = np.where((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r,
                        sh["eps"], cell)
    return cell


def _cells_stack(shape_lists, theta=0.0, phi=0.0):
    st = _stack()
    for sh in shape_lists:
        st.add_layer(_TH, eps_cell=_raster(sh))
    return st.set_source(_WL, theta=theta, phi=phi)


def _solve(st, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = st.solve(**kw).efficiencies()
    return np.asarray(o), np.asarray(R), np.asarray(T)


def _R0(o, R):
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    return R[:, p0]


def _closure(R, T):
    return float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))


@pytest.fixture(scope="module")
def solved():
    """The four solves the physics pins share (module-scoped: ~0.6 s total)."""
    out = {"AB": _solve(_shapes_stack([SH_A, SH_B])),
           "AA": _solve(_shapes_stack([SH_A, SH_A])),
           "AB_oracle": _solve(_cells_stack([SH_A, SH_B])),
           "AA_oracle": _solve(_cells_stack([SH_A, SH_A]))}
    return out


# ---------------------------------------------------------------------------
# 1  key level
# ---------------------------------------------------------------------------

def test_m1_swapped_centre_shape_lists_get_distinct_keys():
    """M1: the two structurally different shape LISTS must not share a key."""
    kA = RCWAStack._layer_eig_key(_shapes_stack([SH_A]).layers[0])
    kB = RCWAStack._layer_eig_key(_shapes_stack([SH_B]).layers[0])
    assert kA != kB, "shapes-layer eig key collides on exchanged centres (M1)"
    assert isinstance(hash(kA), int)      # must stay usable as a dict key
    assert isinstance(hash(kB), int)


def test_m1_identical_shape_lists_still_share_a_key():
    """The dedup must survive: an identical layer keeps ONE key, and the key
    is invariant to the ORDER the shape dict's own fields were written in."""
    kA = RCWAStack._layer_eig_key(_shapes_stack([SH_A]).layers[0])
    kA2 = RCWAStack._layer_eig_key(_shapes_stack([list(SH_A)]).layers[0])
    assert kA == kA2
    shuffled = [{"eps": s["eps"], "center": s["center"], "shape": s["shape"],
                 "radius": s["radius"]} for s in SH_A]
    assert RCWAStack._layer_eig_key(
        _shapes_stack([shuffled]).layers[0]) == kA


def test_m1_repeated_shapes_layer_solves_its_eigenproblem_once(monkeypatch):
    """Perf guard: the legitimate cache HIT still fires (a 4x-repeated shapes
    layer = 1 eigensolve), and two distinct layers = 2 (was 1 under M1)."""
    calls = {"n": 0}
    orig = RCWAStack._layer_modes

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(RCWAStack, "_layer_modes", spy)
    _solve(_shapes_stack([SH_A] * 4), symmetry=False)
    assert calls["n"] == 1
    calls["n"] = 0
    _solve(_shapes_stack([SH_A, SH_B, SH_A, SH_B]), symmetry=False)
    assert calls["n"] == 2


# ---------------------------------------------------------------------------
# 2  physics level
# ---------------------------------------------------------------------------

def test_m1_ab_stack_is_not_the_aa_stack(solved):
    """M1's signature: the (A, B) stack returned the (A, A) answer exactly."""
    _oAB, RAB, TAB = solved["AB"]
    _oAA, RAA, TAA = solved["AA"]
    assert np.max(np.abs(RAB - RAA)) > 1e-4
    assert np.max(np.abs(TAB - TAA)) > 1e-4


def test_m1_ab_stack_matches_the_raster_eps_cell_oracle(solved):
    """The analytic-shapes path must reproduce the SAME geometry rasterised
    onto a pixel cell (independent code path: FFT convolution, no form
    factors).  Measured max|dR| = 3.7e-5 / max|dT| = 1.9e-4 at S = 481; the
    (A, A) control (which never collided) sits at 1.9e-5 / 6.6e-5."""
    oAB, RAB, TAB = solved["AB"]
    oOR, ROR, TOR = solved["AB_oracle"]
    assert np.array_equal(oAB, oOR)
    assert np.max(np.abs(RAB - ROR)) < 1e-3
    assert np.max(np.abs(TAB - TOR)) < 2e-3
    assert abs(float(_R0(oAB, RAB)[0]) - float(_R0(oOR, ROR)[0])) < 1e-4
    # control: the never-colliding (A, A) stack agrees at the same level, so
    # the tolerance above is the raster-vs-analytic floor, not slack.
    oAA, RAA, TAA = solved["AA"]
    _o2, RAAo, TAAo = solved["AA_oracle"]
    assert np.max(np.abs(RAA - RAAo)) < 1e-3
    assert np.max(np.abs(TAA - TAAo)) < 2e-3
    # ... and the two geometries are genuinely far apart at that scale.
    assert np.max(np.abs(ROR - RAAo)) > 1e-3


def test_m1_energy_closure_on_the_fixed_ab_stack(solved):
    """Closure could never SEE M1 (5.8e-15 while 3.7x wrong); pin that the
    fixed path still closes at the same level."""
    for tag in ("AB", "AA", "AB_oracle"):
        _o, R, T = solved[tag]
        assert _closure(R, T) < 1e-10, tag


def test_m1_oblique_incidence_also_distinguishes_the_two_layers():
    """Off-normal (the even-parity fold is off by construction) -- the same
    collision, and the same oracle agreement."""
    oAB, RAB, TAB = _solve(_shapes_stack([SH_A, SH_B], theta=0.12, phi=0.3))
    _oAA, RAA, _TAA = _solve(_shapes_stack([SH_A, SH_A], theta=0.12, phi=0.3))
    _oo, ROR, TOR = _solve(_cells_stack([SH_A, SH_B], theta=0.12, phi=0.3))
    assert np.max(np.abs(RAB - RAA)) > 1e-4
    assert np.max(np.abs(RAB - ROR)) < 1e-3
    assert np.max(np.abs(TAB - TOR)) < 2e-3
    assert _closure(RAB, TAB) < 1e-10


# ---------------------------------------------------------------------------
# 3  the cache stays a pure, deterministic memoization
# ---------------------------------------------------------------------------

def test_m1_repeat_solve_is_bit_identical():
    """Determinism (the audit's verified-clean property): re-solving the same
    stack -- cold and with every module-level cache warm -- is bit-exact."""
    for kw in ({"symmetry": False}, {"symmetry": "auto"}):
        _o1, R1, T1 = _solve(_shapes_stack([SH_A, SH_B]), **kw)
        _o2, R2, T2 = _solve(_shapes_stack([SH_A, SH_B]), **kw)
        assert np.array_equal(R1, R2) and np.array_equal(T1, T2), kw


def test_m1_shapes_dedup_is_bit_exact_vs_no_dedup(monkeypatch):
    """The shapes-layer dedup must be a pure memoization: forcing a distinct
    key per layer (dedup off) reproduces the deduped solve BIT-for-bit."""
    _o1, R1, T1 = _solve(_shapes_stack([SH_A, SH_B, SH_A]), symmetry=False)
    monkeypatch.setattr(RCWAStack, "_layer_eig_key",
                        staticmethod(lambda L: None))
    _o2, R2, T2 = _solve(_shapes_stack([SH_A, SH_B, SH_A]), symmetry=False)
    assert np.array_equal(R1, R2) and np.array_equal(T1, T2)

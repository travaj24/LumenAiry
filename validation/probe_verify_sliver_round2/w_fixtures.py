"""VERIFY ROUND 2 -- my OWN fixtures, written independently of
``validation/probe_pmmstack_sliver_round2/r_fixtures.py`` and of
``validation/probe_verify_sliver/v_fixtures.py``.

Two families:

* ``BITID`` -- 31 stacks that exercise every ``PMMStack`` path the round-2
  change touches (shared / per-layer / conical / slant / out-of-plane and
  gyrotropic tensors / lossy / absorbing superstrate / sweeps at 1, 2 and 4
  workers / ``prepare()`` / ``stabilize='slices'`` / ``internal_field`` /
  ``layer_absorption`` / tapers whose walls the snap DOES merge).  Each
  returns a dict of arrays; the caller hashes them and records the warning
  set.

* the SLIVER family -- a two-slice staircase whose second slice's walls open
  by ``delta`` of the period, its exact ``delta -> 0`` reference, and the
  unguarded solve.  Built here from scratch (different duty cycle, different
  permittivities and a different period from the fix's O-11 fixture) so the
  thresholds are re-derived on geometry the fix did not tune on, plus the
  fix's own O-11 numbers where the censuses have to be compared row for row.
"""
import os
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps


# ---------------------------------------------------------------- tensors ---
def uniaxial(no, ne, tilt, azim=0.0):
    """``R @ diag(no^2, no^2, ne^2) @ R.T`` -- a rotated uniaxial director.

    ``tilt`` rotates the optic axis OUT of the x-y plane about y, ``azim``
    rotates it IN plane about z.  Built the way an LC device is actually
    built, i.e. only NEARLY symmetric in floats."""
    cz, sz = np.cos(azim), np.sin(azim)
    cy, sy = np.cos(tilt), np.sin(tilt)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    R = Rz @ Ry
    return (R @ np.diag([no ** 2, no ** 2, ne ** 2]).astype(complex) @ R.T)


def gyrotropic(eps0, g):
    M = np.eye(3, dtype=complex) * eps0
    M[0, 1] = 1j * g
    M[1, 0] = -1j * g
    return M


def non_hermitian(eps0, a):
    M = np.eye(3, dtype=complex) * eps0
    M[0, 1] = a
    return M


# ------------------------------------------------------ the sliver family ---
# Deliberately NOT the fix's O-11 numbers: a different period, wavelength,
# angle, duty cycle and permittivity pair.
WP, WWL, WTH = 0.74e-6, 0.53e-6, 0.31
WA, WB = 0.19137, 0.71429
WEH, WEP = 1.96, 6.25
WDZ = 0.21e-6 / 3
NO_SNAP_FRAC = 1e-12


def wsegs(a, b, eh=WEH, ep=WEP):
    return [(a, eh), (b - a, ep), (1.0 - b, eh)]


def wbuild(d, deg, *, mf=None, nsub=1.0, nsup=1.0, th=WTH, nl=2, ep=WEP,
           period=WP, wl=WWL, ffo=None, layer_grids="shared", a0=WA, b0=WB,
           eh=WEH, dz=WDZ, phi=0.0, slant=0.0):
    """``nl`` slices whose walls open by ``d`` (a fraction of a period) in
    total across the stack."""
    kw = {} if ffo is None else dict(far_field_orders=ffo)
    if layer_grids != "shared":
        kw["layer_grids"] = layer_grids
    st = PMMStack(period, n_superstrate=nsup, n_substrate=nsub, degree=deg,
                  min_feature=(period * NO_SNAP_FRAC if mf is None
                               else float(mf)), **kw)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(dz, segments=wsegs(a0 - dd, b0 + dd, eh=eh, ep=ep),
                     slant_angle=slant)
    st.set_source(wl, theta=th, phi=phi)
    return st


def unguarded(st):
    """(orders, R, T, worst) with the guard switched OFF -- the pre-fix code
    path bit for bit.  R/T are the FULL (2, n_orders) arrays, order-sorted."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    R = np.asarray(R)[:, i]
    T = np.asarray(T)[:, i]
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return dict(o=o[i], R=np.real(R), T=np.real(T),
                worst=float(np.max(tot)), J=np.asarray(J))


def shared_move(a, b, *, pol=None):
    """Max |dR|, |dT| over the orders two solves share.  ``pol=None`` scores
    BOTH incident polarizations (what the library's arbiter does); ``pol=1``
    is the campaign's ``err`` convention."""
    c = np.intersect1d(a["o"], b["o"])
    if c.size == 0:
        return None
    ia = np.searchsorted(a["o"], c)
    ib = np.searchsorted(b["o"], c)
    sl = slice(None) if pol is None else slice(pol, pol + 1)
    return float(max(np.abs(a["R"][sl][:, ia] - b["R"][sl][:, ib]).max(),
                     np.abs(a["T"][sl][:, ia] - b["T"][sl][:, ib]).max()))


def classify(e, d):
    """The campaign's fitted-constant-free continuity rule: an answer that
    tracks the physical wall shift to 10x is RIGHT, past 100x it is WRONG."""
    return "wrong" if e > 100.0 * d else "right" if e <= 10.0 * d else "grey"


def prescribed(st):
    """The ``min_feature`` (metres) this stack's refusal would prescribe, and
    the widest manufactured cell it is built from -- read off the library's
    own screen so the probe cannot drift from it."""
    hit = ps._cross_layer_sliver([L[1] for L in st._layers],
                                 float(st.min_feature) / float(st.period))
    if hit is None:
        return None
    return dict(mf=2.0 * hit[3] * float(st.period), w_wide=hit[3],
                w_narrow=hit[0], own=hit[4], n_hit=hit[5])


def snapped(st, mf):
    """The unguarded solve of the same stack on the ``min_feature = mf``
    grid -- the arbiter's own measurement, taken independently here."""
    clone = st._min_feature_clone(float(mf))
    clone._src = dict(st._src)
    return unguarded(clone)


def guarded(st):
    """``('ok', payload)`` / ``('refused', message)`` plus the warning texts
    the guarded solve emitted."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            o, R, T, J = st.solve()
        except ValueError as exc:
            return ("refused", str(exc), [str(w.message) for w in rec])
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return ("ok", dict(o=o[i], R=np.real(np.asarray(R))[:, i],
                       T=np.real(np.asarray(T))[:, i], J=np.asarray(J)),
            [str(w.message) for w in rec])


# ------------------------------------------------------ bit-id fixtures -----
def _w01_single_uniform():
    st = PMMStack(0.6e-6, n_substrate=1.52, degree=10, far_field_orders=9)
    st.add_layer(0.18e-6, eps=2.1)
    st.set_source(0.55e-6, theta=0.0)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w02_binary_grating_oblique():
    st = PMMStack(0.8e-6, n_substrate=1.45, degree=12, far_field_orders=11)
    st.add_layer(0.3e-6, segments=[(0.4, 4.0), (0.6, 1.0)])
    st.set_source(0.65e-6, theta=0.37)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w03_three_layer_non_conforming():
    st = PMMStack(0.9e-6, n_substrate=1.5, degree=10, far_field_orders=11)
    st.add_layer(0.12e-6, segments=[(0.30, 5.0), (0.25, 1.0), (0.45, 2.4)])
    st.add_layer(0.15e-6, segments=[(0.34, 5.0), (0.21, 1.0), (0.45, 2.4)])
    st.add_layer(0.10e-6, eps=2.0)
    st.set_source(0.62e-6, theta=0.21)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w04_bragg_abab_dedupe():
    st = PMMStack(0.7e-6, n_substrate=1.5, degree=8, far_field_orders=9)
    for _ in range(4):
        st.add_layer(0.11e-6, segments=[(0.5, 4.2), (0.5, 1.0)])
        st.add_layer(0.09e-6, segments=[(0.5, 2.1), (0.5, 1.0)])
    st.set_source(0.58e-6, theta=0.15)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w05_lossy_layers():
    st = PMMStack(0.75e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.08e-6, segments=[(0.45, -12.0 + 1.4j), (0.55, 1.0)])
    st.add_layer(0.14e-6, segments=[(0.50, 4.0 + 0.02j), (0.50, 2.25)])
    st.set_source(0.63e-6, theta=0.28)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w06_lossy_substrate():
    st = PMMStack(0.85e-6, n_substrate=1.6 + 0.05j, degree=10,
                  far_field_orders=11)
    st.add_layer(0.2e-6, segments=[(0.38, 6.0), (0.62, 1.0)])
    st.set_source(0.7e-6, theta=0.44)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w07_absorbing_superstrate():
    st = PMMStack(0.8e-6, n_superstrate=1.5 + 0.02j, n_substrate=1.45,
                  degree=10, far_field_orders=9)
    st.add_layer(0.16e-6, segments=[(0.4, 4.5), (0.6, 1.2)])
    st.set_source(0.6e-6, theta=0.19)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w08_conical():
    st = PMMStack(0.9e-6, n_substrate=1.48, degree=10, far_field_orders=9)
    st.add_layer(0.2e-6, segments=[(0.42, 5.5), (0.58, 1.0)])
    st.add_layer(0.13e-6, segments=[(0.46, 5.5), (0.54, 1.0)])
    st.set_source(0.68e-6, theta=0.33, phi=0.62)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w09_slant():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.18e-6, segments=[(0.45, 4.0), (0.55, 1.0)],
                 slant_angle=0.17)
    st.set_source(0.62e-6, theta=0.24)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w10_mixed_slant():
    st = PMMStack(0.85e-6, n_substrate=1.5, degree=8, far_field_orders=9)
    st.add_layer(0.12e-6, segments=[(0.40, 4.0), (0.60, 1.0)],
                 slant_angle=0.10)
    st.add_layer(0.12e-6, segments=[(0.44, 4.0), (0.56, 1.0)],
                 slant_angle=-0.14)
    st.set_source(0.66e-6, theta=0.2)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w11_in_plane_director():
    lc = uniaxial(1.51, 1.72, 0.0, azim=np.pi / 4)
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.22e-6, segments=[(0.45, lc), (0.55, 1.0)])
    st.set_source(0.63e-6, theta=0.27)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w12_out_of_plane_director():
    lc = uniaxial(1.51, 1.72, np.pi / 6)
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.22e-6, segments=[(0.45, lc), (0.55, 1.0)])
    st.set_source(0.63e-6, theta=0.27)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w13_gyrotropic():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.2e-6, segments=[(0.5, gyrotropic(4.0, 0.35)), (0.5, 1.0)])
    st.set_source(0.63e-6, theta=0.2)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w14_non_hermitian():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.2e-6, segments=[(0.5, non_hermitian(4.0, 0.2)), (0.5, 1.0)])
    st.set_source(0.63e-6, theta=0.2)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w15_taper_snap_active():
    """A tapered staircase whose adjacent-slice wall collisions the snap DOES
    merge -- the min_feature path the round-2 arbiter prescribes on."""
    st = PMMStack(0.7e-6, n_substrate=1.5, degree=8, far_field_orders=9,
                  min_feature=1.5e-9)
    st.add_tapered_grating(0.32e-6, eps_ridge=6.25, eps_groove=1.0,
                           duty_bottom=0.4520, duty_top=0.4480, n_slices=8)
    st.set_source(0.6e-6, theta=0.18)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w16_taper_no_snap():
    st = PMMStack(0.7e-6, n_substrate=1.5, degree=8, far_field_orders=9)
    st.add_tapered_grating(0.32e-6, eps_ridge=6.25, eps_groove=1.0,
                           duty_bottom=0.48, duty_top=0.42, n_slices=4)
    st.set_source(0.6e-6, theta=0.18)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w17_per_layer_window():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=8, far_field_orders=9,
                  layer_grids="per-layer")
    for k in range(5):
        a = 0.30 + 0.011 * k
        st.add_layer(0.06e-6, segments=[(a, 5.0), (0.42, 1.0),
                                        (1.0 - a - 0.42, 2.2)])
    st.set_source(0.61e-6, theta=0.23)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w18_per_layer_window_hw2():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=8, far_field_orders=9,
                  layer_grids="per-layer", window_halfwidth=2)
    for k in range(6):
        a = 0.30 + 0.013 * k
        st.add_layer(0.05e-6, segments=[(a, 5.0), (0.40, 1.0),
                                        (1.0 - a - 0.40, 2.2)])
    st.set_source(0.61e-6, theta=0.23)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w19_stabilize_slices():
    st = PMMStack(0.7e-6, n_substrate=1.5, degree=8, far_field_orders=9,
                  min_feature=1.0e-9)
    st.add_tapered_grating(0.30e-6, eps_ridge=6.25, eps_groove=1.0,
                           duty_bottom=0.47, duty_top=0.43, n_slices=5)
    st.set_source(0.6e-6, theta=0.2)
    o, R, T, J = st.solve(stabilize="slices")
    return dict(o=o, R=R, T=T, J=J)


def _w20_internal_field():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.12e-6, segments=[(0.40, 5.0), (0.60, 1.0)])
    st.add_layer(0.15e-6, segments=[(0.44, 5.0), (0.56, 1.0)])
    st.set_source(0.62e-6, theta=0.22)
    st.solve(retain_internal=True)
    f = st.internal_field(0.14e-6, nx=24)
    return {k: np.asarray(v) for k, v in f.items()}


def _w21_layer_absorption():
    st = PMMStack(0.75e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.09e-6, segments=[(0.45, -14.0 + 1.6j), (0.55, 1.0)])
    st.add_layer(0.16e-6, segments=[(0.50, 4.0 + 0.03j), (0.50, 2.25)])
    st.set_source(0.63e-6, theta=0.26)
    o, R, T, J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    return dict(o=o, R=R, T=T, J=J, A=np.asarray(A))


def _w22_per_order_amplitudes():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=11)
    st.add_layer(0.2e-6, segments=[(0.42, 5.0), (0.58, 1.0)])
    st.set_source(0.64e-6, theta=0.3)
    o, R, T, J = st.solve()
    ar = st.per_order_amplitudes(port="reflection")
    at = st.per_order_amplitudes(port="transmission")
    return dict(o=o, R=R, T=T, J=J,
                ar_Ex=np.asarray(ar["Ex"]), ar_Ey=np.asarray(ar["Ey"]),
                ar_kz=np.asarray(ar["kz"]),
                at_Ex=np.asarray(at["Ex"]), at_Ey=np.asarray(at["Ey"]),
                at_kz=np.asarray(at["kz"]))


def _sweep(nw):
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=8, far_field_orders=9)
    st.add_layer(0.12e-6, segments=[(0.40, 5.0), (0.60, 1.0)])
    st.add_layer(0.15e-6, segments=[(0.45, 5.0), (0.55, 1.0)])
    st.set_source(4.0e-7, theta=0.2)          # deliberately STALE
    wl = np.array([6.0e-7, 6.5e-7, 7.0e-7, 8.5e-7])
    o, R, T, J = st.solve_vs_wavelength(wl, theta=0.2, jones=True,
                                        max_workers=nw)
    return dict(o=o, R=R, T=T, J=J)


def _w23_sweep_w1():
    return _sweep(1)


def _w24_sweep_w2():
    return _sweep(2)


def _w25_sweep_w4():
    return _sweep(4)


def _w26_prepare():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    st.add_layer(0.2e-6, segments=[(0.45, "LC"), (0.55, 1.0)])
    prep = st.prepare()
    out = {}
    for j, azim in enumerate((0.0, np.pi / 3)):
        o, R, T, J = prep.solve(wavelength=6.3e-7, angle=0.24,
                                materials={"LC": uniaxial(1.51, 1.72, 0.0,
                                                          azim=azim)})
        out[f"o{j}"], out[f"R{j}"] = np.asarray(o), np.asarray(R)
        out[f"T{j}"], out[f"J{j}"] = np.asarray(T), np.asarray(J)
    return out


def _w27_sweep_dispersive():
    st = PMMStack(0.8e-6, n_substrate=1.5, degree=8, far_field_orders=9)
    st.add_layer(0.15e-6, segments=[(0.42, lambda w: 5.0 + 2e5 * w),
                                    (0.58, 1.0)])
    st.set_source(6.0e-7, theta=0.2)
    o, R, T, J = st.solve_vs_wavelength(np.array([6.0e-7, 7.0e-7]),
                                        theta=0.2, jones=True, max_workers=1)
    return dict(o=o, R=R, T=T, J=J)


def _w28_sliver_below_trigger():
    """A stack that DOES carry a manufactured sliver but whose answer closes
    -- the guard must be inert and the numbers must not move."""
    st = wbuild(3e-3, 10)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _superunity_stack(deg, th, nsup):
    """A DENSE-superstrate grazing mount whose walls are IDENTICAL in every
    layer -- so the union manufactures NO cell and the guard's geometric
    screen cannot fire -- but which is under-converged enough to read
    super-unity.  The plain warning here must be word-for-word what the
    pre-round-2 tip emits."""
    a0, b0 = 0.27865, 0.62505
    st = PMMStack(1.2e-6, n_superstrate=nsup, n_substrate=1.5 + 0.2j,
                  degree=deg, far_field_orders=31)
    for _ in range(2):
        st.add_layer(0.08e-6, segments=[(a0, 2.25), (b0 - a0, 9.0),
                                        (1.0 - b0, 2.25)])
    st.set_source(0.85e-6, theta=th)
    return st


def _w29_superunity_no_sliver():
    """R+T = 1.0389, no manufactured cell: the PLAIN super-unity warning, and
    round 2 must not append a syllable to it."""
    st = _superunity_stack(6, 1.2, 2.5)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w30_between_trigger_and_bar():
    """R+T = 1.00225 -- ABOVE round 2's new trigger and BELOW the warning bar,
    with no sliver.  Round 2 lowered the trigger, so this is the fixture that
    proves the lowered trigger alone creates no new report."""
    st = _superunity_stack(8, 1.2, 2.5)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


def _w31_benign_within_layer_liner():
    """A 1e-3-of-a-period liner ONE layer owns, on a converged solve: the
    within-layer arm must stay silent."""
    st = PMMStack(1.2e-6, n_substrate=1.5, degree=12, far_field_orders=21)
    st.add_layer(0.08e-6, segments=[(0.30, 2.25), (1e-3, 9.0),
                                    (0.70 - 1e-3, 2.25)])
    st.add_layer(0.08e-6, segments=[(0.30, 2.25), (1e-3, 9.0),
                                    (0.70 - 1e-3, 2.25)])
    st.set_source(0.85e-6, theta=0.15)
    o, R, T, J = st.solve()
    return dict(o=o, R=R, T=T, J=J)


BITID = {
    "w01_single_uniform": _w01_single_uniform,
    "w02_binary_grating_oblique": _w02_binary_grating_oblique,
    "w03_three_layer_non_conforming": _w03_three_layer_non_conforming,
    "w04_bragg_abab_dedupe": _w04_bragg_abab_dedupe,
    "w05_lossy_layers": _w05_lossy_layers,
    "w06_lossy_substrate": _w06_lossy_substrate,
    "w07_absorbing_superstrate": _w07_absorbing_superstrate,
    "w08_conical": _w08_conical,
    "w09_slant": _w09_slant,
    "w10_mixed_slant": _w10_mixed_slant,
    "w11_in_plane_director": _w11_in_plane_director,
    "w12_out_of_plane_director": _w12_out_of_plane_director,
    "w13_gyrotropic": _w13_gyrotropic,
    "w14_non_hermitian": _w14_non_hermitian,
    "w15_taper_snap_active": _w15_taper_snap_active,
    "w16_taper_no_snap": _w16_taper_no_snap,
    "w17_per_layer_window": _w17_per_layer_window,
    "w18_per_layer_window_hw2": _w18_per_layer_window_hw2,
    "w19_stabilize_slices": _w19_stabilize_slices,
    "w20_internal_field": _w20_internal_field,
    "w21_layer_absorption": _w21_layer_absorption,
    "w22_per_order_amplitudes": _w22_per_order_amplitudes,
    "w23_sweep_w1": _w23_sweep_w1,
    "w24_sweep_w2": _w24_sweep_w2,
    "w25_sweep_w4": _w25_sweep_w4,
    "w26_prepare": _w26_prepare,
    "w27_sweep_dispersive": _w27_sweep_dispersive,
    "w28_sliver_below_trigger": _w28_sliver_below_trigger,
    "w29_superunity_no_sliver": _w29_superunity_no_sliver,
    "w30_between_trigger_and_bar": _w30_between_trigger_and_bar,
    "w31_benign_within_layer_liner": _w31_benign_within_layer_liner,
}

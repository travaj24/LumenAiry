"""V12 -- did the branch-cut fix remove the M1 guard's motivating population?

BACKGROUND.  ``_core._guarded_inverse`` screens every explicit inverse on
``_rcond_1_equilibrated`` and, where a site ARMS it, refuses on
``_equilibrated_inverse_residual``.  The whole reason it scores the
EQUILIBRATED operator rather than the raw one is a measured false positive:
``tests/unit/test_m1_conditioning_guard.py::
test_anisotropic_cascade_is_not_falsely_refused`` records a uniaxial
``rcwa_jones_1d`` cascade whose RAW inverse residual runs 1e-02 .. 5e-01 at
every truncation -- which a raw-residual bar would refuse -- while the
equilibrated residual reads 1e-15 .. 6e-14 and the answer is right.

That cell's layer background is ``1.5^2`` and ``n_substrate`` is 1.5: it is a
COINCIDENCE cell, and the coordinator's reading is that its raw residual was
the branch-cut defect all along.  If so, the fix removes the population that
motivated the instrument, and the test's premise (b) -- "the raw instruments
would refuse every rung" -- stops being true.

THIS PROBE ASKS THREE THINGS, on both arms in one process:

  1. the failing test's own ladder (M = 5, 9, 15, 19, 25), with the raw and
     equilibrated instruments read off every guarded inverse the solve
     performs, at whatever thread count the caller pinned;
  2. whether ANY fixture still produces a MOTIVATING call -- one where the raw
     residual exceeds ``_INV_RESID_REFUSE`` (so a raw bar would refuse) while
     the equilibrated residual is below it (so equilibration rescues a correct
     answer).  That is the population the instrument exists for, and if it is
     empty post-fix the instrument has no measured justification left;
  3. whether the ARMED site (`rcwa generalized interface (T22)`, the only
     caller that passes ``rcond_refuse``) still sees anything near its bar.

The sweep covers the M1 cell over five truncations and two mounts, plus the
2-D anisotropic coincidence cell, the uniform-spacer stack, the 1-D groove
case, a metal, a loss ladder and the off-coincidence controls -- 24 solves.

Usage:  OPENBLAS_NUM_THREADS=<n> python v12_m1_instrument.py <out.json>
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402

_MODULES = ("lumenairy.elements.rcwa._core", "lumenairy.elements.pmm._core")


class GuardSpy:
    """Record the M1 instruments on every guarded inverse the solve performs.

    Delegates to the shipped function, so the solve is unchanged -- the same
    shape the M1 test's own spy uses.
    """

    def __init__(self):
        self.rows = []
        self._saved = []

    def __enter__(self):
        from lumenairy.elements.rcwa import _core as rc
        orig = rc._guarded_inverse
        rows = self.rows

        def spy(A, site, hint=None, rcond_refuse=None):
            A_np = np.asarray(A)
            if (A_np.ndim == 2 and A_np.shape[0] == A_np.shape[1]
                    and np.all(np.isfinite(A_np))):
                try:
                    X = np.linalg.inv(A_np)
                    rows.append(dict(
                        site=site, n=int(A_np.shape[0]),
                        raw=float(rc._inverse_residual(A_np, X)),
                        eq=float(rc._equilibrated_inverse_residual(A_np)),
                        rcond_eq=float(rc._rcond_1_equilibrated(A_np, X)),
                        armed=(None if rcond_refuse is None
                               else float(rcond_refuse))))
                except np.linalg.LinAlgError:            # pragma: no cover
                    rows.append(dict(site=site, n=int(A_np.shape[0]),
                                     raw=float("inf"), eq=float("inf"),
                                     rcond_eq=0.0, armed=rcond_refuse))
            return orig(A, site, hint, rcond_refuse) if rcond_refuse is not None \
                else orig(A, site, hint)

        for name in _MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:                            # pragma: no cover
                continue
            if hasattr(mod, "_guarded_inverse"):
                self._saved.append((mod, mod._guarded_inverse))
                mod._guarded_inverse = spy
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._guarded_inverse = fn
        return False


# ------------------------------------------------------------------ fixtures
WL = 700e-9


def _m1_cell(M, angle_deg=10.0):
    from lumenairy.elements.rcwa import rcwa_jones_1d, uniaxial_tensor
    er = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.deg2rad(20))
    eg = (1.5 ** 2) * np.eye(3)
    return rcwa_jones_1d(1.0e-6, er, eg, 1.5, 1.0, 0.4e-6, 0.5, WL,
                         angle=np.deg2rad(angle_deg), n_orders=M)


def _m1_cell_detuned(M):
    """The SAME cell with the coincidence removed: groove 1.5^2 kept, substrate
    walked to 1.63.  If the raw residual is the branch-cut defect, this row
    should look post-fix-like on BOTH arms."""
    from lumenairy.elements.rcwa import rcwa_jones_1d, uniaxial_tensor
    er = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.deg2rad(20))
    eg = (1.5 ** 2) * np.eye(3)
    return rcwa_jones_1d(1.0e-6, er, eg, 1.63, 1.0, 0.4e-6, 0.5, WL,
                         angle=np.deg2rad(10), n_orders=M)


def fixtures():
    from lumenairy.elements.rcwa import RCWAStack, rcwa_efficiency_1d, rcwa_jones_1d, rcwa_jones_2d
    F = []
    for M in (5, 9, 15, 19, 25):
        F.append(("m1_aniso_M%d" % M, lambda M=M: _m1_cell(M)))
    for M in (5, 15, 25):
        F.append(("m1_aniso_DETUNED_M%d" % M, lambda M=M: _m1_cell_detuned(M)))
    F.append(("m1_aniso_M15_normal", lambda: _m1_cell(15, angle_deg=0.0)))
    F.append(("jones2d_coinc", lambda: rcwa_jones_2d(
        V._P, V._P, V.uniaxial_cell(), 1.5, 1.0, V._DEPTH, V._WL,
        n_orders_x=5, n_orders_y=5)))
    F.append(("jones2d_offcoinc", lambda: rcwa_jones_2d(
        V._P, V._P, V.uniaxial_cell(), 1.63, 1.0, V._DEPTH, V._WL,
        n_orders_x=5, n_orders_y=5)))
    F.append(("oned_groove_eq_sub_te", lambda: rcwa_efficiency_1d(
        V._P, 2.1, 1.5, 1.5, 1.0, V._DEPTH, 0.5, V._WL, polarization="te",
        n_orders=15)))
    F.append(("oned_metal_tm", lambda: rcwa_efficiency_1d(
        V._P, 0.2 + 3.4j, 1.5, 1.5, 1.0, V._DEPTH, 0.5, V._WL,
        polarization="tm", n_orders=15)))
    for im in (1e-2, 1e-6):
        F.append(("lossy_im%.0e" % im, lambda im=im: rcwa_jones_2d(
            V._P, V._P, V.uniaxial_cell(eps_im=im), 1.5, 1.0, V._DEPTH, V._WL,
            n_orders_x=5, n_orders_y=5)))
    F.append(("jones1d_conical_coinc", lambda: rcwa_jones_1d(
        V._P, np.diag([4.41 + 0j] * 3), np.diag([2.25 + 0j] * 3), 1.5, 1.0,
        V._DEPTH, 0.5, V._WL, theta=0.3, n_orders=15)))

    def _spacer(n_sub, spacer_eps):
        st = RCWAStack(period=V._P, period_y=V._P, n_superstrate=1.0,
                       n_substrate=n_sub, n_orders=3, n_orders_y=3)
        st.add_layer(0.05e-6, eps=spacer_eps)
        st.add_layer(0.12e-6, eps_tensor_cell=V.uniaxial_cell(S=24))
        st.add_layer(0.06e-6, eps=spacer_eps)
        return st.set_source(V._WL).solve()
    F.append(("stack_spacer_coinc", lambda: _spacer(1.63, 2.25)))
    F.append(("stack_spacer_control", lambda: _spacer(1.63, 2.56)))

    # --- the ARMED site.  ``_guarded_inverse``'s refusal is dormant everywhere
    # except ``_interface_smatrix_general``'s T22 inverse, which is reached
    # only by the GENERALIZED cascade: an out-of-plane tensor (eps_xz != 0)
    # and Berreman.  Without these the sweep never sees the armed bar at all.
    def _oop_cell(tilt=0.6, bg=2.25):
        tc = V.uniaxial_cell(twist=0.0, bg=bg)
        no2, ne2 = 2.25, 2.89
        c, sn = np.cos(tilt), np.sin(tilt)
        x = (np.arange(48) + 0.5) / 48 - 0.5
        m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
        tc[m, 0, 0] = ne2 * c * c + no2 * sn * sn
        tc[m, 2, 2] = ne2 * sn * sn + no2 * c * c
        tc[m, 0, 2] = tc[m, 2, 0] = (ne2 - no2) * c * sn
        return tc
    F.append(("oop_2d_coinc", lambda: rcwa_jones_2d(
        V._P, V._P, _oop_cell(), 1.5, 1.0, V._DEPTH, V._WL, n_orders_x=4,
        n_orders_y=4)))
    F.append(("oop_2d_offcoinc", lambda: rcwa_jones_2d(
        V._P, V._P, _oop_cell(), 1.63, 1.0, V._DEPTH, V._WL, n_orders_x=4,
        n_orders_y=4)))

    def _oop_1d(n_sub):
        er = np.array([[2.89, 0.0, 0.5], [0.0, 2.25, 0.0],
                       [0.5, 0.0, 2.25]], dtype=complex)
        eg = (1.5 ** 2) * np.eye(3, dtype=complex)
        return rcwa_jones_1d(1.0e-6, er, eg, n_sub, 1.0, 0.4e-6, 0.5, WL,
                             angle=np.deg2rad(10), n_orders=15)
    F.append(("oop_1d_coinc", lambda: _oop_1d(1.5)))
    F.append(("oop_1d_offcoinc", lambda: _oop_1d(1.63)))

    def _berr(n_sub):
        from lumenairy.elements.berreman import berreman_jones_1d
        eps = np.diag([2.89 + 0j, 2.25 + 0j, 2.25 + 0j])
        return berreman_jones_1d([(eps, 0.2e-6), (2.25, 0.1e-6)], n_sub, 1.0,
                                 V._WL, angle=0.3)
    F.append(("berreman_coinc", lambda: _berr(1.5)))
    F.append(("berreman_offcoinc", lambda: _berr(1.63)))
    return F


REFUSE = 1e-8       # _INV_RESID_REFUSE
SCREEN = 1e-8       # _INV_RCOND_SCREEN
T22_BAR = 1e-10     # _INV_T22_RCOND_REFUSE


def measure(call):
    spy = GuardSpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with spy:
            try:
                call()
                err = None
            except Exception as exc:
                err = type(exc).__name__
    rows = spy.rows
    if not rows:
        return dict(error=err, n_calls=0)
    raw = [r["raw"] for r in rows]
    eq = [r["eq"] for r in rows]
    rc = [r["rcond_eq"] for r in rows]
    # MOTIVATING: a raw bar would refuse, equilibration rescues it.
    motiv = [r for r in rows if r["raw"] > REFUSE and r["eq"] <= REFUSE]
    armed = [r for r in rows if r["armed"] is not None]
    return dict(
        error=err, n_calls=len(rows),
        raw_max=max(raw), eq_max=max(eq), rcond_min=min(rc),
        ratio_raw_over_eq=(max(raw) / max(max(eq), 1e-300)),
        n_motivating=len(motiv),
        worst_motivating_raw=(max(r["raw"] for r in motiv) if motiv else None),
        n_armed_calls=len(armed),
        armed_rcond_min=(min(r["rcond_eq"] for r in armed) if armed else None),
        armed_raw_max=(max(r["raw"] for r in armed) if armed else None),
        armed_eq_max=(max(r["eq"] for r in armed) if armed else None),
        armed_below_bar=sum(1 for r in armed if r["rcond_eq"] < T22_BAR),
        sites=sorted({r["site"] for r in rows}),
    )


def main():
    V.require_local_tree()
    out = sys.argv[1]
    res = {}
    for name, call in fixtures():
        post = measure(call)
        with V.PreSqrtDecay():
            pre = measure(call)
        res[name] = dict(post=post, pre=pre)
        print("%-24s PRE raw=%-10s eq=%-10s rcond=%-10s motiv=%-3s | "
              "POST raw=%-10s eq=%-10s rcond=%-10s motiv=%s" % (
                  name,
                  "%.3e" % pre.get("raw_max", float("nan")),
                  "%.3e" % pre.get("eq_max", float("nan")),
                  "%.3e" % pre.get("rcond_min", float("nan")),
                  pre.get("n_motivating", "-"),
                  "%.3e" % post.get("raw_max", float("nan")),
                  "%.3e" % post.get("eq_max", float("nan")),
                  "%.3e" % post.get("rcond_min", float("nan")),
                  post.get("n_motivating", "-")))
    tot_pre = sum(v["pre"].get("n_motivating", 0) for v in res.values())
    tot_post = sum(v["post"].get("n_motivating", 0) for v in res.values())
    calls_pre = sum(v["pre"].get("n_calls", 0) for v in res.values())
    calls_post = sum(v["post"].get("n_calls", 0) for v in res.values())
    print("")
    print("MOTIVATING CALLS (raw > %g AND eq <= %g): PRE %d of %d, "
          "POST %d of %d" % (REFUSE, REFUSE, tot_pre, calls_pre, tot_post,
                             calls_post))
    armed_pre = min((v["pre"]["armed_rcond_min"] for v in res.values()
                     if v["pre"].get("armed_rcond_min") is not None),
                    default=None)
    armed_post = min((v["post"]["armed_rcond_min"] for v in res.values()
                      if v["post"].get("armed_rcond_min") is not None),
                     default=None)
    print("ARMED SITE (T22) equilibrated rcond, minimum over the sweep: "
          "PRE %s POST %s (refusal bar %g)"
          % ("%.3e" % armed_pre if armed_pre is not None else "-",
             "%.3e" % armed_post if armed_post is not None else "-", T22_BAR))
    V.dump(out, dict(fixtures=res, totals=dict(
        motivating_pre=tot_pre, motivating_post=tot_post,
        calls_pre=calls_pre, calls_post=calls_post,
        armed_rcond_min_pre=armed_pre, armed_rcond_min_post=armed_post),
        openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS",
                                            "unpinned")))


if __name__ == "__main__":
    main()

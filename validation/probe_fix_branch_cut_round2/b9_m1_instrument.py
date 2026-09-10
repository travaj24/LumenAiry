"""B9 -- does the M1 equilibrated-residual instrument still have a motivating
population?

``rcwa/_core._guarded_inverse`` screens every explicit inverse with
``_rcond_1_equilibrated`` and, below the screen, scores it with
``_equilibrated_inverse_residual`` rather than with the RAW residual.  The
EQUILIBRATED instrument was chosen because a population existed where the raw
one would have REFUSED a correct answer: that is what "the false positive the
equilibration exists for" means in
``docs/audits/PMM_M1_CONDITIONING_2026_08_04.md``.

A call is MOTIVATING when the raw residual would refuse it
(``raw > _INV_RESID_REFUSE``) and the equilibrated one rescues it
(``eq <= _INV_RESID_REFUSE``).  This probe counts that population on both arms
over a sweep that includes the M1 cascade itself, its detuned control, the X-1
ladder, the 2-D anisotropic coincidence, the uniform-spacer stacks of round 2,
a loss ladder and the surfaces that reach the ARMED ``T22`` site.

The decision this feeds is NOT whether to delete the instrument -- round 2
removes no library code -- but whether the justification recorded in the tests
still describes a state that occurs.

Usage: OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b9_m1_instrument.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
WLX = 700e-9
HOST = 2.25
WEAK = HOST * (1.0 + 1e-6)


class InvSpy:
    """Score the RAW and the EQUILIBRATED instruments on every guarded inverse
    the solve performs, delegating to the shipped function so nothing about the
    solve changes."""

    def __init__(self):
        self.rows = []
        self._saved = None

    def __enter__(self):
        from lumenairy.elements.rcwa import _core as rc
        orig = rc._guarded_inverse
        rows = self.rows

        def spy(A, site, hint=None):
            A_np = np.asarray(A)
            if (A_np.ndim == 2 and A_np.shape[0] == A_np.shape[1]
                    and np.all(np.isfinite(A_np))):
                try:
                    X = np.linalg.inv(A_np)
                    rows.append((site,
                                 float(rc._inverse_residual(A_np, X)),
                                 float(rc._equilibrated_inverse_residual(A_np)),
                                 float(rc._rcond_1_equilibrated(A_np, X))))
                except Exception:                         # pragma: no cover
                    pass
            return orig(A, site, hint)

        self._saved = (rc, orig)
        rc._guarded_inverse = spy
        return self

    def __exit__(self, *a):
        mod, fn = self._saved
        mod._guarded_inverse = fn
        return False


def fixtures():
    from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_efficiency_2d_cell
    from lumenairy.elements.rcwa import (
        rcwa_efficiency_1d,
        rcwa_jones_1d,
        rcwa_jones_2d,
        uniaxial_tensor,
    )
    out = []
    er = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.deg2rad(20))
    eg = (1.5 ** 2) * np.eye(3)
    for M in (5, 9, 15, 19, 25):
        out.append(("m1_cascade_M%d" % M,
                    lambda m=M: rcwa_jones_1d(1.0e-6, er, eg, 1.5, 1.0,
                                              0.4e-6, 0.5, WLX,
                                              angle=np.deg2rad(10),
                                              n_orders=m)))
    out.append(("m1_cascade_normal",
                lambda: rcwa_jones_1d(1.0e-6, er, eg, 1.5, 1.0, 0.4e-6, 0.5,
                                      WLX, angle=0.0, n_orders=15)))
    for M in (5, 15, 25):
        out.append(("m1_detuned_M%d" % M,
                    lambda m=M: rcwa_jones_1d(1.0e-6, er, eg, 1.63, 1.0,
                                              0.4e-6, 0.5, WLX,
                                              angle=np.deg2rad(10),
                                              n_orders=m)))
    for M in (12, 19, 20, 21, 28):
        for pol in ("te", "tm"):
            out.append(("x1_%d_%s" % (M, pol),
                        lambda m=M, p=pol: rcwa_efficiency_1d(
                            10e-6, 1.55, 1.5, 1.5, 1.5, 0.5e-6, 0.5, WLX,
                            angle=0.0, polarization=p, n_orders=m)))
    out.append(("aniso2d_coinc",
                lambda: rcwa_jones_2d(PX, PX, F.tensor_cell(S=32), 1.5, 1.0,
                                      D, WL, n_orders_x=4, n_orders_y=4,
                                      symmetry=False)))
    out.append(("aniso2d_lossy",
                lambda: rcwa_jones_2d(PX, PX, F.tensor_cell(S=32, eps_im=1e-2),
                                      1.5, 1.0, D, WL, n_orders_x=4,
                                      n_orders_y=4, symmetry=False)))

    def spacer_stack(nsub):
        st = PMM2DStackHybrid(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                              degree=7, n_orders=4, symmetry=False)
        st.add_layer(0.1e-6, eps=HOST)
        c = np.full((6, 6), HOST + 0j)
        c[2:4, 2:4] = WEAK
        st.add_layer(D, eps_cell=c)
        st.add_layer(0.1e-6, eps=HOST)
        return st.set_source(WL, theta=0.0).solve()

    out.append(("spacer_stack_off", lambda: spacer_stack(1.63)))
    out.append(("spacer_stack_coinc", lambda: spacer_stack(1.5)))
    c = np.full((6, 6), HOST + 0j)
    c[2:4, 2:4] = 6.0
    out.append(("pmm_cell", lambda: pmm_efficiency_2d_cell(
        PX, PX, c, 1.5, 1.0, D, WL, degree=7, n_orders=4, symmetry=False)))
    for ei in (1e-2, 1e-6):
        out.append(("groove_lossy_%g" % ei,
                    lambda e=ei: rcwa_efficiency_1d(
                        1.0e-6, (2.1 + 1j * e) ** 0.5, 1.5, 1.5, 1.0, 0.4e-6,
                        0.5, WL, polarization="tm", n_orders=15)))
    out.append(("metal_1d", lambda: rcwa_efficiency_1d(
        1.0e-6, (-10 + 1j) ** 0.5, 1.0, 1.5, 1.0, 0.1e-6, 0.5, WL,
        polarization="tm", n_orders=11)))
    out.append(("oop_tensor", lambda: rcwa_jones_2d(
        PX, PX, _oop_cell(), 1.5, 1.0, D, WL, n_orders_x=3, n_orders_y=3,
        symmetry=False)))
    return out


def _oop_cell(S=24):
    tc = F.tensor_cell(S=S)
    x = (np.arange(S) + 0.5) / S - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    tc[m, 0, 2] = tc[m, 2, 0] = 0.4
    return tc


def sweep():
    from lumenairy.elements.rcwa import _core as rc
    rows = []
    for name, fn in fixtures():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with InvSpy() as sp:
                try:
                    fn()
                except Exception:
                    pass
        for site, raw, eq, rcond in sp.rows:
            rows.append(dict(fixture=name, site=site, raw=raw, eq=eq,
                             rcond=rcond,
                             motivating=bool(raw > rc._INV_RESID_REFUSE
                                             and eq <= rc._INV_RESID_REFUSE)))
    return rows


def summarise(rows):
    from lumenairy.elements.rcwa import _core as rc
    mot = [r for r in rows if r["motivating"]]
    armed = [r for r in rows if "T22" in r["site"]]
    return dict(
        n_calls=len(rows), n_motivating=len(mot),
        motivating_fixtures=sorted({r["fixture"] for r in mot}),
        max_raw=max((r["raw"] for r in rows), default=None),
        max_eq=max((r["eq"] for r in rows), default=None),
        max_ratio=max((r["raw"] / max(r["eq"], 1e-300) for r in rows),
                      default=None),
        n_armed=len(armed),
        armed_min_rcond=min((r["rcond"] for r in armed), default=None),
        refuse_bar=rc._INV_RESID_REFUSE)


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b9.json"
    post = sweep()
    with F.PreSqrtDecayRCWA():
        pre = sweep()
    res = dict(post=summarise(post), pre=summarise(pre))
    for arm in ("pre", "post"):
        s = res[arm]
        print("%-4s calls=%-4d motivating=%-3d maxRaw=%-11s maxEq=%-11s "
              "maxRaw/Eq=%-11s armed=%-3d armedMinRcond=%s"
              % (arm, s["n_calls"], s["n_motivating"],
                 "%.3e" % s["max_raw"] if s["max_raw"] is not None else "-",
                 "%.3e" % s["max_eq"] if s["max_eq"] is not None else "-",
                 "%.3e" % s["max_ratio"] if s["max_ratio"] is not None
                 else "-",
                 s["n_armed"],
                 "%.3e" % s["armed_min_rcond"]
                 if s["armed_min_rcond"] is not None else "-"))
        if s["motivating_fixtures"]:
            print("       motivating fixtures: "
                  + ", ".join(s["motivating_fixtures"]))
    F.dump(out, dict(summary=res, pre_rows=pre, post_rows=post,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()

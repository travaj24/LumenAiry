# CALIBRATION of the v5.32 ray-density HALO self-check
# (``_RD_HALO_AMAX_TOL`` / ``_RD_HALO_RADIUS_FACTOR`` in
# ``lumenairy/elements/_lens_traced.py``).
#
# THE RULE THIS ENFORCES.  A guard that would have refused its own shipped
# configuration is worse than no guard.  So the bound is not chosen, it is
# fitted to two measured populations:
#
#   CLEAN      -- configurations that are currently accepted AND independently
#                 shown correct: the CI-safe P2 design battery (four designs x
#                 two beam sizes x two aperture:beam ratios, gated in
#                 tests/unit/test_niche_p2_design_battery.py against an exact
#                 meridional ray oracle), the five synthetic C6 ghost fixtures
#                 that are clean on both branches, and design 121's own fan on
#                 every configuration whose halo is under its exact-ray ceiling.
#   DEFECTIVE  -- configurations carrying a lobe CONFIRMED manufactured against
#                 an exact ray trace (design 121's on-axis C6 call with the fit
#                 guard off; the fit guard's own regression on two synthetic
#                 fixtures; the C5-off + C6-on interaction at the extreme
#                 tilted orders).
#
# HOW THE READING IS OBTAINED.  ``_RD_HALO_AMAX_TOL`` is forced negative so the
# check fires on EVERY call and prints its own numbers, which include the
# exact-ray support radius and the traced exit centroid.  Those are captured
# together with the returned field, so ``amax_halo`` at ANY radius factor is
# recomputed script-side from one pass -- the factor sweep costs nothing extra
# and no library constant is swept.
#
# LOCAL-ONLY: constants are set as module attributes inside try/finally.
#
# usage:  PART=synth  python halo_calibration.py
#         PART=batt   python halo_calibration.py
#         PART=d121   ORDERS='0,0 -4,-2' python halo_calibration.py
import os
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lumenairy as la                                            # noqa: E402,I001
import lumenairy.elements as _EL                                  # noqa: E402
import lumenairy.elements._lens_traced as LT                      # noqa: E402

FACTORS = (1.00, 1.10, 1.25, 1.50, 2.00)

_RX = re.compile(
    r"amax_halo = ([0-9.e+-]+) of peak beyond ([0-9.e+-]+) mm "
    r"\(([0-9.]+) x the exact-ray exit support radius ([0-9.e+-]+) mm "
    r"about the traced exit centroid \(([0-9.e+-]+), ([0-9.e+-]+)\) mm\)")


class capture(object):
    """Record every ``apply_real_lens_traced`` call's input, output, dx and the
    halo self-check's own reading (support radius + traced exit centroid)."""

    def __init__(self):
        self.calls = []
        self._orig = None

    def __enter__(self):
        self._orig = _EL.apply_real_lens_traced

        def _w(E_in, *a, **kw):
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                out = self._orig(E_in, *a, **kw)
            rec = {'dx': float(kw.get('dx', a[-1] if a else np.nan)),
                   'E_out': np.array(out, copy=True),
                   'p_in': float((np.abs(np.asarray(E_in)) ** 2).sum()),
                   'hull': None, 'energy': 0}
            for w in wl:
                t = str(w.message)
                if 'energy self-check FAILED' in t:
                    rec['energy'] += 1
                m = _RX.search(t)
                if m is not None:
                    rec['hull'] = {'r_hull': float(m.group(4)) * 1e-3,
                                   'cx': float(m.group(5)) * 1e-3,
                                   'cy': float(m.group(6)) * 1e-3}
            self.calls.append(rec)
            return out

        _EL.apply_real_lens_traced = _w
        return self

    def __exit__(self, *e):
        _EL.apply_real_lens_traced = self._orig
        return False


def _rec_from(E_in, out, dx, wl):
    rec = {'dx': float(dx), 'E_out': np.asarray(out),
           'p_in': float((np.abs(np.asarray(E_in)) ** 2).sum()),
           'hull': None, 'energy': 0}
    for w in wl:
        t = str(w.message)
        if 'energy self-check FAILED' in t:
            rec['energy'] += 1
        m = _RX.search(t)
        if m is not None:
            rec['hull'] = {'r_hull': float(m.group(4)) * 1e-3,
                           'cx': float(m.group(5)) * 1e-3,
                           'cy': float(m.group(6)) * 1e-3}
    return rec


def score(rec):
    """``amax_halo`` and ``g_halo`` at every radius factor, recomputed from the
    captured field and the check's own support radius."""
    h = rec['hull']
    if h is None:
        return None
    F = rec['E_out']
    n = F.shape[0]
    ax = (np.arange(n) - n / 2) * rec['dx']
    r2 = (ax[None, :] - h['cx']) ** 2 + (ax[:, None] - h['cy']) ** 2
    aF = np.abs(F)
    pk = float(aF.max())
    out = {'r_hull': h['r_hull']}
    for f in FACTORS:
        m = r2 > (f * h['r_hull']) ** 2
        if not m.any() or pk <= 0.0:
            out[f] = (float('nan'), float('nan'))
        else:
            out[f] = (float(aF[m].max()) / pk,
                      float((aF[m] ** 2).sum()) / rec['p_in'])
    return out


def hdr():
    h = f"{'fixture':<44} {'r_hull/mm':>9} {'P/Pin':>9} " + ' '.join(
        f"{'x%.2f' % f:>10}" for f in FACTORS) + f" {'g@1.25':>10}"
    print(h)
    print('-' * len(h))


def row(lbl, rec):
    s = score(rec)
    p = float((np.abs(rec['E_out']) ** 2).sum()) / rec['p_in']
    if s is None:
        print(f"{lbl:<44} {'--':>9} {p:>9.5f}   (hull beyond the grid: the "
              f"check is vacuous on this call)")
        return
    cells = ' '.join(f"{s[f][0]:>10.3e}" for f in FACTORS)
    print(f"{lbl:<44} {s['r_hull'] * 1e3:>9.4f} {p:>9.5f} {cells} "
          f"{s[1.25][1]:>10.3e}", flush=True)


# ---------------------------------------------------------------------------
# (A) the synthetic C6 ghost fixtures
# ---------------------------------------------------------------------------
def part_synth():
    import probe_ghost_synthetic as GS                            # noqa: I001
    hdr()
    for (lbl, n, dx, w, rc, al, r1, r2, th, z, ap) in GS.CASES:
        E, _X, _Y = GS.field(n, dx, w, rc, al)
        presc = GS.singlet(r1, r2, th, z, ap)
        for blab, launch, guard in (('C6off', False, False),
                                    ('mask ', True, False),
                                    ('wght ', True, True)):
            # GS.run calls ``la.apply_real_lens_traced`` directly, so the
            # element monkeypatch cannot see it -- record the warnings here.
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                F = GS.run(E, dx, presc, rc, launch, guard)
            row(f"{lbl[:36]:<36} {blab}", _rec_from(E, F, dx, wl))
        print()


# ---------------------------------------------------------------------------
# (B) the CI-safe P2 design battery -- the "must never be refused" population
# ---------------------------------------------------------------------------
def part_batt():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests',
        'unit'))
    import test_niche_p2_design_battery as B                      # noqa: I001
    rs = int(os.environ.get('RS', str(B._RS)))
    print(f"   P2 battery at N={B._N}, ray_subsample={rs}")
    hdr()
    for (name, design, w0, ratio) in B._CELLS:
        ap = ratio * 2.0 * w0
        gwg = design(ap)
        groups = [{'prescription': p, 'gap_before': g} for (p, g) in gwg]
        env0, dx = B._launch(ap, w0)
        with capture() as cap:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                la.propagate_traced_carrier_chain(
                    env0, groups, B._WL, dx, r_in=np.inf, ray_subsample=rs,
                    n_workers=1, final_distance=0.0,
                    traced_kwargs=dict(parallel_amp=False,
                                       on_undersample='silent'))
        for k, rec in enumerate(cap.calls):
            row(f"{name}-w{w0 * 1e3:g}mm-ap{ratio:g}x rs{rs} grp{k}", rec)
        print()


# ---------------------------------------------------------------------------
# (C) design 121's own fan
# ---------------------------------------------------------------------------
def part_d121():
    import _d121_common as C                                      # noqa: I001
    from lumenairy.elements._lens_traced import TiltedCarrier     # noqa: I001
    LAM = C.LAM
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get('ORDERS', '0,0 -2,0 -4,-2').split()]
    cfgs = {'ship': (True, True, False), 'shipG': (True, True, True),
            'noC6': (True, False, False), 'noC5': (False, True, False)}
    names = [c.strip() for c in
             os.environ.get('CFGS', 'ship,shipG,noC6').split(',') if c.strip()]
    rs = int(os.environ.get('RS', '4'))
    _pre, post, _g, period = C.geometry()
    env_doe, R_doe, dx_doe, _P = C.chain_a(n=1024, rs=rs)
    hdr()
    for order in orders:
        L, M = order[0] * LAM / period, order[1] * LAM / period
        for nm in names:
            c5, c6, gd = cfgs[nm]
            old = (LT.TILTED_CARRIER_EXACT_EIKONAL,
                   LT.REMAP_STATIONARY_PHASE_LAUNCH,
                   LT.REMAP_STATIONARY_PHASE_FIT_GUARD)
            (LT.TILTED_CARRIER_EXACT_EIKONAL,
             LT.REMAP_STATIONARY_PHASE_LAUNCH,
             LT.REMAP_STATIONARY_PHASE_FIT_GUARD) = (c5, c6, gd)
            try:
                with capture() as cap:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        la.propagate_traced_carrier_chain(
                            env_doe, [dict(g) for g in post], LAM, dx_doe,
                            r_in=TiltedCarrier(R_doe, L, M), ray_subsample=rs,
                            n_workers=8, final_distance=0.0,
                            final_leg='paraxial', on_decentred_fit='ignore')
            finally:
                (LT.TILTED_CARRIER_EXACT_EIKONAL,
                 LT.REMAP_STATIONARY_PHASE_LAUNCH,
                 LT.REMAP_STATIONARY_PHASE_FIT_GUARD) = old
            for k, rec in enumerate(cap.calls):
                row(f"121 {str(order):<8} {nm:<6} grp{k}", rec)
            print()


def main():
    import hashlib
    print("   lib %s  sha256 %s" % (
        os.path.basename(LT.__file__),
        hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]))
    print(f"   halo check: contour e^-{LT._RD_HALO_AMP_CONTOUR:g}, shipped "
          f"factor {LT._RD_HALO_RADIUS_FACTOR:g}, shipped tol "
          f"{LT._RD_HALO_AMAX_TOL:.1e}")
    print("   columns xF = amax_halo (max |E| beyond F x the exact-ray exit "
          "support radius, over peak).")
    print("   g@1.25 = the POWER fraction beyond 1.25x, reported for context "
          "and NOT part of the bar.")
    print()
    old = LT._RD_HALO_AMAX_TOL
    LT._RD_HALO_AMAX_TOL = -1.0        # fire on every call so it prints
    try:
        part = os.environ.get('PART', 'synth')
        {'synth': part_synth, 'batt': part_batt, 'd121': part_d121}[part]()
    finally:
        LT._RD_HALO_AMAX_TOL = old


if __name__ == '__main__':
    sys.exit(main())

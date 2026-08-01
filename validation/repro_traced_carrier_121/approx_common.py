# Shared harness for the 2026-07-30 APPROXIMATION AUDIT of the traced-carrier
# path (docs/audits/APPROXIMATION_AUDIT_TRACED_2026_07_30.md).
#
# LOCAL-ONLY (needs the 121 .zmx + the design-study runner).  Nothing here is
# library code and nothing here EDITS library code: every "exact alternative"
# is installed as a RUNTIME monkeypatch on a module attribute and removed in a
# ``finally``, so the shipped files are untouched.
#
# The instrument is a DIFFERENTIAL one: the same end-to-end chain (chain A
# cached -> TiltedCarrier(order) -> the six post-DOE groups -> the trailing leg
# -> the EXACT Bluestein readout) is run once per variant on a FIXED output
# lattice, and every number is quoted as a delta against the shipped-default
# baseline run in the SAME process.  A "null" variant (identity patch) is run
# first to establish the differential floor; a delta below that floor is not a
# measurement.
#
# Why a fixed lattice: ABLATE_LAST_GROUP_2026_07_30 S6 showed that a hand-off
# taken mid-leg is not a valid measurement.  This harness never hands off --
# it only changes ONE construction inside a complete shipped run and reads the
# same image plane, so there is no hand-off plane to be wrong about.
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings('ignore', message='.*prescription aperture.*')
warnings.filterwarnings('ignore', message='.*residual transverse.*')
warnings.filterwarnings('ignore', message='.*under-sampled.*')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- PIN the library ------------------------------------------------------
# ``lumenairy/elements/_lens_traced.py`` is being edited by another agent
# WHILE this audit runs (niche C6).  A moving target is not auditable and a
# half-saved file crashed the first batch, so every number in this study is
# taken against a frozen ``git archive HEAD`` export of the package.  Set
# LUMEN_PIN=0 to measure the live working tree instead.
PIN = os.environ.get(
    'LUMEN_PIN',
    r"C:\Users\Tesla\AppData\Local\Temp\claude\C--Users-Tesla"
    r"\372a2d1f-acbe-4b57-a148-eeae3fe1d729\scratchpad\pin_d2e60ca")
if PIN and PIN != '0' and os.path.isdir(os.path.join(PIN, 'lumenairy')):
    sys.path.insert(0, PIN)

# import the PACKAGE first, so it is resolved from the pin and lands in
# sys.modules before _d121_common prepends the live repo root.
import lumenairy as la                                         # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402
import lumenairy.elements as EL                                # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402

import _d121_common as C                                       # noqa: E402

LIB_FILE = LT.__file__

LAM = C.LAM
K0 = 2.0 * np.pi / LAM
TRAILING = C.TRAILING

RN = int(os.environ.get('RN', 1024))
RS = int(os.environ.get('RS', 4))
NW = int(os.environ.get('NW', 8))
NFC = int(os.environ.get('NFC', 12288))
WF = float(os.environ.get('WF', 4.0))
NOUT = int(os.environ.get('NOUT', 192))
DXO = float(os.environ.get('DXO', 0.1)) * 1e-6


class Patch(object):
    """Context manager installing a list of ``(module, attr, value)``."""

    def __init__(self, items):
        self.items = list(items)
        self._old = []

    def __enter__(self):
        for mod, attr, val in self.items:
            self._old.append((mod, attr, getattr(mod, attr)))
            setattr(mod, attr, val)
        return self

    def __exit__(self, *a):
        for mod, attr, old in reversed(self._old):
            setattr(mod, attr, old)
        self._old = []
        return False


def setup(order):
    """Return ``(post, env, R, dx, P_in, L, M, centre_out)`` for one order."""
    m, n = order
    _pre, post, _gap, period = C.geometry()
    env, R, dx, P_in = C.chain_a(n=RN)
    L, M = m * LAM / period, n * LAM / period
    car = la.TiltedCarrier(R, L, M)
    xc, yc, _L, _M = CM._chain_chief_ray_at_target(
        post, LAM, car, TRAILING, 'approx_common')
    return post, env, R, dx, P_in, L, M, (float(xc), float(yc))


def run_chain(post, env, R, dx, L, M, centre_out, chain_kw=None,
              traced_kw=None, patches=()):
    """One complete shipped-path run.  Returns ``(field, stages, secs)``."""
    ck = dict(ray_subsample=RS, n_workers=NW, final_distance=TRAILING,
              final_leg='exact', on_decentred_fit='ignore',
              on_gap_paraxial='ignore', on_na_proximity='ignore')
    fr = {'dx_out': DXO, 'N_out': NOUT, 'n_fine_cap': NFC,
          'window_factor': WF, 'centre_out': centre_out}
    if chain_kw:
        fr.update(chain_kw.pop('focus_readout', {}) or {})
        ck.update(chain_kw)
    ck['focus_readout'] = fr
    if traced_kw:
        ck['traced_kwargs'] = dict(traced_kw)
    t0 = time.time()
    with Patch(patches):
        res = la.propagate_traced_carrier_chain(
            env, post, LAM, dx, r_in=la.TiltedCarrier(R, L, M), **ck)
    return np.asarray(res.field), res.stages, time.time() - t0


def metrics(E, P_in, dxo=None):
    """Peak-centred encircled energy + tile power.  ``EE*`` normalised to the
    chain's INPUT power so variants that lose energy are visible."""
    dxo = DXO if dxo is None else dxo
    I = np.abs(E) ** 2
    n = E.shape[-1]
    iy, ix = np.unravel_index(np.argmax(I), I.shape)
    xx = (np.arange(n) - ix) * dxo
    yy = (np.arange(n) - iy) * dxo
    rr = np.hypot(xx[None, :], yy[:, None])
    out = {'P_tile': float(I.sum()) * dxo * dxo / P_in,
           'peak': float(I[iy, ix]),
           'off_x': float((ix - n // 2) * dxo),
           'off_y': float((iy - n // 2) * dxo)}
    for r in (3, 6, 12):
        out['EE%d' % r] = float(I[rr <= r * 1e-6].sum()) * dxo * dxo / P_in
    return out


def field_diff(E, E0):
    """Relative L2 difference and the amplitude-weighted rms phase difference
    (rad) over the pixels holding 99.9 % of the baseline power."""
    d = np.linalg.norm(E - E0) / max(np.linalg.norm(E0), 1e-300)
    I0 = np.abs(E0) ** 2
    fl = np.sort(I0.ravel())[::-1]
    cs = np.cumsum(fl)
    thr = fl[np.searchsorted(cs, 0.999 * cs[-1])] if cs[-1] > 0 else 0.0
    m = I0 >= thr
    if not m.any():
        return float(d), np.nan
    u = (E * np.conj(E0))[m]
    ph = np.angle(u * np.conj(np.sum(u * I0[m]) /
                             max(abs(np.sum(u * I0[m])), 1e-300)))
    w = I0[m] / I0[m].sum()
    return float(d), float(np.sqrt(np.sum(w * ph ** 2)))


def report(rows):
    hdr = ("%-34s %7s %7s %7s %8s %10s %9s %6s" %
           ('variant', 'EE3%', 'EE6%', 'EE12%', 'Ptile%', 'relL2', 'dphi_rms',
            's'))
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        if r.get('err'):
            print("%-34s  FAILED: %s" % (r['name'], r['err'][:90]))
            continue
        print("%-34s %7.3f %7.3f %7.3f %8.4f %10.3e %9.2e %6.0f" %
              (r['name'], r['EE3'] * 100, r['EE6'] * 100, r['EE12'] * 100,
               r['P_tile'] * 100, r['relL2'], r['dphi'], r['secs']))

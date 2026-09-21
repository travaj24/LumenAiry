"""WP-C3 round 3, probe 2.

(a) the matched (frac = 1.00) row of the a6 C1 fixture on both transports,
    against the CLOSED-FORM analytic focal peak  (w_in/w0)^2;
(b) the downward-quadratic fixture's gated (short) leg, field accuracy on
    both transports against the analytic ABCD field of the COMPOSED
    curvature;
(c) certification that the analytic ABCD width IS the exact second-moment
    law  <r^2>(z) = <r^2> + 2 z <r.theta> + z^2 <theta^2>  -- the law is read
    off the ENVELOPE (which the grid resolves) and the carrier, a known pure
    quadratic, is composed analytically;
(d) which public entry points reach the readout, and whether one keyword is
    the way back at each.

    python c3d_a6_measure2.py <tree> <out.json>
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import pathlib
import platform
import sys
import warnings

import numpy as np

TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, TREE)
import lumenairy                                            # noqa: E402
from lumenairy.propagators import carrier as C              # noqa: E402

assert os.path.abspath(lumenairy.__file__).lower().startswith(TREE.lower())
print('[bind] lumenairy.__file__ =', lumenairy.__file__, flush=True)
_DEF = inspect.signature(
    C.carrier_referenced_focus_readout).parameters['transport'].default
print('[bind] readout transport default =', _DEF, flush=True)

OUT = {'tree': TREE, 'lumenairy': lumenairy.__file__,
       'python': sys.version.split()[0], 'numpy': np.__version__,
       'platform': platform.platform(), 'readout_default_transport': _DEF}
TR = ('sziklas', 'collins')


def grid(n, dx):
    return (np.arange(n, dtype=np.float64) - n / 2) * dx


def abcd_field(xo, w_in, r_beam, z, lam):
    kk = 2.0 * np.pi / lam
    inv_q = 1.0 / r_beam + 1j * lam / (np.pi * w_in ** 2)
    q = 1.0 / inv_q
    q2 = q + z
    wz = float(np.sqrt(lam / (np.pi * np.imag(1.0 / q2))))
    r2 = xo[None, :] ** 2 + xo[:, None] ** 2
    return (np.exp(1j * kk * z) * (q / q2)
            * np.exp(1j * kk * r2 / (2.0 * q2))), wz


def grade(f, truth):
    pist = np.angle(np.vdot(truth, f))
    rel = float(np.linalg.norm(f * np.exp(-1j * pist) - truth)
                / np.linalg.norm(truth))
    peak = float((np.abs(f) ** 2).max() / (np.abs(truth) ** 2).max())
    return peak, rel


def ro(env, R, z, lam, dx, tr, **kw):
    pd = {}
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        try:
            f = C.carrier_referenced_focus_readout(
                env, R, z, lam, dx, transport=tr, _period_out=pd, **kw)
        except Exception as exc:                            # noqa: BLE001
            return None, type(exc).__name__, pd, []
    return f, None, pd, [str(x.message)[:90] for x in wl]


# ---------------------------------------------------------------------------
# (c) the second-moment law, read off the envelope with the carrier composed
# ---------------------------------------------------------------------------
def moment_law_from_envelope(u, dx, lam, R, z):
    """Exact free-space second-moment law.

    ``u`` is the ENVELOPE on the input plane, referenced to the pure
    quadratic carrier ``exp(i k r^2 / 2R)``; the carrier's contribution to
    the moments is composed in closed form, so nothing here differentiates or
    transforms an under-sampled phase.  Returns the 1/e AMPLITUDE radius
    ``w(z) = sqrt(2 <r^2>(z))`` -- the convention
    ``_envelope_amp_radius`` uses.
    """
    u = np.asarray(u)
    n = u.shape[-1]
    x = grid(n, dx)
    k = 2.0 * np.pi / lam
    I = np.abs(u) ** 2
    P = float(I.sum())
    r2 = float((I * (x[None, :] ** 2 + x[:, None] ** 2)).sum()) / P
    # envelope's own <r.theta> and <theta^2>
    F = np.fft.fft2(u)
    S = np.abs(F) ** 2
    fx = np.fft.fftfreq(n, dx)
    th2_u = float((S * (lam * lam)
                   * (fx[None, :] ** 2 + fx[:, None] ** 2)).sum()) / float(S.sum())
    gy, gx = np.gradient(u, dx, dx)
    rt_u = float(np.imag(
        (np.conj(u) * (x[None, :] * gx + x[:, None] * gy)).sum())) / P / k
    # compose the carrier (theta -> theta + r/R)
    invR = 0.0 if not np.isfinite(R) else 1.0 / float(R)
    rt = rt_u + r2 * invR
    th2 = th2_u + 2.0 * rt_u * invR + r2 * invR * invR
    r2z = r2 + 2.0 * z * rt + z * z * th2
    return float(np.sqrt(2.0 * max(r2z, 0.0)))


def sec_c_moment_certification():
    """Cross-check: for a Gaussian the moment law and the ABCD width agree."""
    rows = []
    for lam, w_in, r0, ext, n in ((1.31e-6, 1.0e-3, -20e-3, 4.0, 1024),
                                  (0.85e-6, 0.6e-3, -7.5e-3, 4.0, 1024),
                                  (0.85e-6, 200e-6, -20e-3, 6.0, 512)):
        dx = 2.0 * ext * w_in / n
        g = grid(n, dx)
        r2 = g[None, :] ** 2 + g[:, None] ** 2
        u = np.exp(-r2 / w_in ** 2).astype(complex)       # FLAT envelope
        for frac in (0.02, 0.2, 0.5, 0.9, 0.97):
            z = frac * abs(r0)
            w_law = moment_law_from_envelope(u, dx, lam, r0, z)
            _, w_abcd = abcd_field(np.zeros(1), w_in, r0, z, lam)
            rows.append(dict(lam=lam, w_in=w_in, r0=r0, ext=ext, z=z,
                             w_law_um=w_law * 1e6, w_abcd_um=w_abcd * 1e6,
                             rel=abs(w_law - w_abcd) / w_abcd))
    return rows


# ---------------------------------------------------------------------------
# (a) the matched row of the a6 C1 fixture
# ---------------------------------------------------------------------------
def sec_a_matched():
    LAM = 1.31e-6
    K0 = 2.0 * np.pi / LAM
    n, w_in, na, ext = 1024, 1.0e-3, 0.05, 4.0
    r0 = -w_in / na
    dx = 2.0 * ext * w_in / n
    g = grid(n, dx)
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    e_phys = np.exp(-r2 / w_in ** 2) * np.exp(1j * K0 * r2 / (2.0 * r0))
    w0 = LAM * abs(r0) / (np.pi * w_in)
    z = -r0
    xo = grid(64, w0 / 8.0)
    truth, wz = abcd_field(xo, w_in, r0, z, LAM)
    out = {'w0_closed_form_um': w0 * 1e6, 'w_abcd_at_focus_um': wz * 1e6,
           'analytic_peak_intensity': (w_in / w0) ** 2,
           'abcd_peak_intensity': float((np.abs(truth) ** 2).max()),
           'moment_law_at_focus_um': None}
    u_flat = np.exp(-r2 / w_in ** 2).astype(complex)
    out['moment_law_at_focus_um'] = moment_law_from_envelope(
        u_flat, dx, LAM, r0, z) * 1e6
    env = C.carrier_referenced_envelope(e_phys, r0, LAM, dx)
    for tr in TR:
        f, exc, pd, wl = ro(env, r0, z, LAM, dx, tr,
                            dx_out=w0 / 8.0, N_out=64, on_replica='ignore')
        if exc:
            out[tr] = {'raised': exc}
            continue
        pk, rel = grade(f, truth)
        out[tr] = {'peak_vs_abcd': pk, 'relL2_vs_abcd': rel,
                   'containment': pd.get('containment'),
                   'standoff_um': pd.get('standoff', 0.0) * 1e6,
                   'warnings': wl}
    return out


# ---------------------------------------------------------------------------
# (b) the downward-quadratic gated leg, field accuracy
# ---------------------------------------------------------------------------
def sec_b_gated_field():
    lam = 0.85e-6
    k = 2.0 * np.pi / lam
    n, w2, ext, r = 512, 200e-6, 6.0, -20e-3
    inv_res = -60.0
    dx = 2.0 * ext * w2 / n
    g = grid(n, dx)
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    env = np.exp(-r2 / w2 ** 2) * np.exp(1j * k * r2 * 0.5 * inv_res)
    z = -r
    r_eff = 1.0 / (1.0 / r + inv_res)
    xo = grid(48, 1e-6)
    truth, wz = abcd_field(xo, w2, r_eff, z, lam)
    u_flat = np.exp(-r2 / w2 ** 2).astype(complex)
    out = {'r_eff_mm': r_eff * 1e3, 'w_abcd_at_target_um': wz * 1e6,
           'moment_law_at_target_um': moment_law_from_envelope(
               u_flat, dx, lam, r_eff, z) * 1e6}
    real = C._beam_containment_standoff

    def _gated(*a, **kw):
        r_ = a[1]
        zeta_cf = -float(r_)
        w_, half_ = a[5], a[7]
        iv = kw.get('inv_env')
        iv = 0.0 if iv is None else float(iv)
        cc = 1.0 / float(r_) + iv
        zr_ = np.pi * w_ * w_ / a[3]
        al = (half_ ** 2 / zeta_cf ** 2
              - (C._FOCUS_STANDOFF_MARGIN * w_) ** 2
              * (cc * cc + 1.0 / zr_ ** 2))
        return real(*a, **kw) if al > 0.0 else 0.0

    for tag, patch in (('resolved', False), ('gated', True)):
        try:
            if patch:
                C._beam_containment_standoff = _gated
            for tr in TR:
                f, exc, pd, wl = ro(env, r, z, lam, dx, tr, dx_out=1e-6,
                                    N_out=48, on_replica='ignore',
                                    on_focus_containment='ignore')
                d = {'raised': exc}
                if exc is None:
                    pk, rel = grade(f, truth)
                    d.update(peak_vs_abcd=pk, relL2_vs_abcd=rel,
                             containment=pd.get('containment'),
                             containment_model=pd.get('containment_model'),
                             standoff_um=pd.get('standoff', 0.0) * 1e6)
                out[f'{tag}_{tr}'] = d
                # default disposition
                f2, exc2, pd2, _ = ro(env, r, z, lam, dx, tr, dx_out=1e-6,
                                      N_out=48, on_replica='ignore')
                out[f'{tag}_{tr}_default'] = exc2 or 'returned'
        finally:
            C._beam_containment_standoff = real
    return out


# ---------------------------------------------------------------------------
# (d) entry-point reachability
# ---------------------------------------------------------------------------
def sec_d_entrypoints():
    root = pathlib.Path(C.__file__).parents[1]
    hits = []
    for p in sorted(root.rglob('*.py')):
        try:
            tree = ast.parse(p.read_text(encoding='utf-8', errors='replace'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            nm = (f.id if isinstance(f, ast.Name)
                  else f.attr if isinstance(f, ast.Attribute) else None)
            if nm != 'carrier_referenced_focus_readout':
                continue
            kws = {kw.arg for kw in node.keywords}
            hits.append({'file': str(p.relative_to(root)), 'line': node.lineno,
                         'names_transport': 'transport' in kws,
                         'splat': None in kws})
    # GUI docks
    gui = []
    for p in sorted(root.rglob('*.py')):
        t = p.read_text(encoding='utf-8', errors='replace')
        if 'focus_readout' in t and 'gui' in str(p).replace('\\', '/'):
            gui.append(str(p.relative_to(root)))
    return {'call_sites': hits, 'gui_files_mentioning_focus_readout': gui,
            'chain_transport_default': inspect.signature(
                C.propagate_traced_carrier_chain
            ).parameters['transport'].default,
            'multi_transport_default': inspect.signature(
                C.propagate_traced_carrier_chain_multi
            ).parameters['transport'].default}


if __name__ == '__main__':
    OUT['c_moment_certification'] = sec_c_moment_certification()
    print('c done', flush=True)
    OUT['a_matched'] = sec_a_matched()
    print('a done', flush=True)
    OUT['b_gated'] = sec_b_gated_field()
    print('b done', flush=True)
    OUT['d_entrypoints'] = sec_d_entrypoints()
    print('d done', flush=True)
    with open(sys.argv[2], 'w', encoding='utf-8') as fh:
        json.dump(OUT, fh, indent=1, default=str)
    print('WROTE', sys.argv[2])

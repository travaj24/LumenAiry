"""VERIFY-WP-C3 ROUND 2, item 2 (D8) -- the focus readout's new ``transport``
keyword, re-measured independently.

Round 2 replaced WP-C3's hard pin of the standoff leg to ``'sziklas'`` with a
public ``transport`` keyword whose DEFAULT is ``'sziklas'``.  The claim that
justifies the keyword is a number: on this function's own 128-grid fixture the
Collins standoff leg reads relative L2 ``4.7340e-05`` against a converged
dense separable Fresnel oracle and the co-moving one ``2.4049``.

This probe re-derives that with its OWN oracle, and it drives the SHIPPED
keyword (``transport=``) rather than monkey-patching
``propagate_carrier_referenced``, so it also checks that the keyword actually
reaches the leg.

Two oracles, deliberately different:

  O1  a dense separable Fresnel quadrature of the ANALYTIC input, on a
      composite Simpson rule at 512x the input pitch (the input
      ``exp(-r^2/w^2) exp(i k r^2 / 2R)`` is separable, so the 2-D integral is
      an outer product of 1-D ones, exactly).  Convergence is reported
      256x -> 512x.
  O2  the UNTRUNCATED analytic ABCD Gaussian at the readout plane.  The window
      truncates the fixture's Gaussian at 2.13 w (1.05 % in amplitude), so O2
      cannot confirm a 5e-05 residual -- it is here to bound the answer's
      SCALE independently of any quadrature.

Run with cwd = the tree root, PYTHONPATH = the tree root,
VC3_TREE = the tree root, VC3_OUT = the output directory, VC3_TAG = a tag.
"""
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(os.environ['VC3_TREE'])
sys.path.insert(0, TREE)
import lumenairy                                              # noqa: E402
import lumenairy.propagators.carrier as C                     # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)

N, DX, WL = 128, 4e-6, 633e-9
W, R, Z, STANDOFF = 120e-6, -0.03, 0.03, 1e-3
DXO, NOUT = 2e-7, 32
K = 2.0 * np.pi / WL


def env_of(n, dx, w):
    x = (np.arange(n) - n / 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(np.complex128)


def simpson_w(m, h):
    """Composite Simpson weights on m samples (m odd)."""
    w = np.ones(m)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * h / 3.0


def fresnel_1d_dense(x0, x1, w, r, xo, z, wl, over):
    """1-D Fresnel integral of exp(-x^2/w^2) exp(i k x^2 / 2r) by Simpson."""
    k = 2.0 * np.pi / wl
    m = over + 1
    xs = np.linspace(x0, x1, m)
    ws = simpson_w(m, (x1 - x0) / (m - 1))
    es = np.exp(-xs ** 2 / w ** 2) * np.exp(1j * k * xs ** 2 / (2.0 * r))
    integ = es * np.exp(1j * k * xs ** 2 / (2.0 * z)) * ws
    return np.exp(-1j * k * np.outer(xo, xs) / z) @ integ


def oracle_dense(n, dx, w, r, z, wl, xo, over):
    x = (np.arange(n) - n / 2) * dx
    x0, x1 = float(x[0]), float(x[-1] + dx)
    f = fresnel_1d_dense(x0, x1, w, r, xo, z, wl, over)
    k = 2.0 * np.pi / wl
    pre = np.exp(1j * k * z) / (1j * wl * z)
    return pre * np.outer(f, f) * np.exp(
        1j * k * (xo[:, None] ** 2 + xo[None, :] ** 2) / (2.0 * z))


def oracle_abcd(w, r, z, wl, xo):
    q = 1.0 / complex(1.0 / r, wl / (np.pi * w * w))
    inv = 1.0 / (q + z)
    w_out = float(np.sqrt(wl / (np.pi * inv.imag)))
    rr = (xo[:, None] ** 2 + xo[None, :] ** 2)
    amp = np.exp(-rr / w_out ** 2)
    return amp.astype(np.complex128), w_out


def compare(f, T):
    if f is None:
        return None
    f = np.asarray(f)
    ov = np.vdot(T, f)
    sc = ov / np.vdot(T, T)
    ph = ov / abs(ov) if abs(ov) > 0 else 1.0
    return {'relL2': float(np.linalg.norm(f - T) / np.linalg.norm(T)),
            'relL2_phase_free': float(
                np.linalg.norm(f / ph - T) / np.linalg.norm(T)),
            'relL2_scale_free': float(
                np.linalg.norm(f / sc - T) / np.linalg.norm(T)),
            'best_scale_abs': float(abs(sc)),
            'peak_ratio': float(np.abs(f).max() ** 2 / np.abs(T).max() ** 2)}


def readout(env, r, z, dx, tr, n=N, **extra):
    kw = dict(dx_out=DXO, N_out=NOUT, standoff=STANDOFF, on_replica='ignore',
              transport=tr)
    kw.update(extra)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            f = C.carrier_referenced_focus_readout(env, r, z, WL, dx, **kw)
        return np.asarray(f), [str(w.message)[:120] for w in rec], None
    except Exception as exc:                                  # noqa: BLE001
        return None, [], type(exc).__name__ + ': ' + str(exc)[:200]


def main():
    import inspect
    out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__}
    sig = inspect.signature(C.carrier_referenced_focus_readout).parameters
    out['readout_has_transport_kw'] = 'transport' in sig
    out['readout_transport_default'] = (
        sig['transport'].default if 'transport' in sig else None)

    xo = (np.arange(NOUT) - NOUT / 2) * DXO
    o256 = oracle_dense(N, DX, W, R, Z, WL, xo, 256 * N)
    o512 = oracle_dense(N, DX, W, R, Z, WL, xo, 512 * N)
    out['oracle_256x_vs_512x_relL2'] = float(
        np.linalg.norm(o256 - o512) / np.linalg.norm(o512))
    ORACLE = o512
    abcd, w_out = oracle_abcd(W, R, Z, WL, xo)
    out['analytic_w_out_um'] = w_out * 1e6
    out['oracle_vs_abcd_amp_relL2'] = float(
        np.linalg.norm(np.abs(ORACLE) / np.abs(ORACLE).max()
                       - np.abs(abcd) / np.abs(abcd).max())
        / np.linalg.norm(np.abs(abcd) / np.abs(abcd).max()))

    env = env_of(N, DX, W)
    out['fixture'] = dict(N=N, dx=DX, wl=WL, w=W, R=R, z=Z,
                          standoff=STANDOFF, dx_out=DXO, N_out=NOUT)
    rows = {}
    for tr in ('sziklas', 'collins'):
        f, wr, err = readout(env, R, Z, DX, tr)
        rows[tr] = {'raised': err, 'warnings': wr,
                    'vs_oracle': compare(f, ORACLE)}
        f2, wr2, err2 = readout(env, R, Z, DX, tr,
                                on_focus_containment='warn')
        rows[tr + '_guard_warn'] = {'raised': err2, 'warnings': wr2,
                                    'vs_oracle': compare(f2, ORACLE)}
    out['headline'] = rows

    # the 5 geometries x 6 standoffs sweep
    geoms = [dict(w=120e-6, R=-0.03, z=0.03, N=128, dx=4e-6),
             dict(w=200e-6, R=-0.05, z=0.05, N=128, dx=8e-6),
             dict(w=80e-6, R=-0.02, z=0.02, N=128, dx=3e-6),
             dict(w=150e-6, R=-0.04, z=0.04, N=256, dx=4e-6),
             dict(w=300e-6, R=-0.08, z=0.08, N=256, dx=10e-6)]
    stands = (0.2e-3, 0.5e-3, 1e-3, 2e-3, 4e-3, 8e-3)
    sweep = []
    for gi, g in enumerate(geoms):
        e = env_of(g['N'], g['dx'], g['w'])
        for so in stands:
            xo_g = (np.arange(NOUT) - NOUT / 2) * DXO
            orc = oracle_dense(g['N'], g['dx'], g['w'], g['R'], g['z'], WL,
                               xo_g, 256 * g['N'])
            rec = {'geom': gi, 'standoff_mm': so * 1e3}
            for tr in ('sziklas', 'collins'):
                f, wr, err = readout(e, g['R'], g['z'], g['dx'], tr,
                                     standoff=so)
                rec[tr] = {'raised': (err.split(':')[0] if err else None),
                           'relL2': (None if f is None
                                     else compare(f, orc)['relL2'])}
            sweep.append(rec)
    out['sweep'] = sweep
    nr = {tr: sum(1 for r in sweep if r[tr]['raised']) for tr in
          ('sziklas', 'collins')}
    out['sweep_raise_counts'] = nr
    for tr in ('sziklas', 'collins'):
        vals = sorted(r[tr]['relL2'] for r in sweep
                      if r[tr]['relL2'] is not None)
        out['sweep_relL2_' + tr] = {
            'n': len(vals), 'max': (vals[-1] if vals else None),
            'n_le_5p3e-4': sum(1 for v in vals if v <= 5.3e-4),
            'n_le_3.846e-3': sum(1 for v in vals if v <= 3.846e-3)}

    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.environ['VC3_OUT'], f'vr2_d8_standoff_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, default=repr)
    print('transport kw present:', out['readout_has_transport_kw'],
          '| default =', out['readout_transport_default'])
    print('oracle 256x vs 512x relL2 =', out['oracle_256x_vs_512x_relL2'])
    print('oracle vs untruncated ABCD (amp, normalised) =',
          out['oracle_vs_abcd_amp_relL2'])
    for k, v in rows.items():
        c = v['vs_oracle']
        print(f"  {k:22s} raised={v['raised']}  "
              f"relL2={None if c is None else c['relL2']!r}  "
              f"scale={None if c is None else c['best_scale_abs']!r}  "
              f"peak_ratio={None if c is None else c['peak_ratio']!r}")
    print('sweep raises (of 30):', nr)
    for tr in ('sziklas', 'collins'):
        print('  ', tr, out['sweep_relL2_' + tr])
    print('WROTE', p)


if __name__ == '__main__':
    main()

"""VERIFY-WP-C5 item 1 -- the near-focus exact-kernel switch, RE-MEASURED.

Independent of ``validation/probe_c5_three_defaults/c5_item1_kernel.py``: the
two published fixtures are rebuilt from their own definitions (so their numbers
can be refuted), and THREE fixtures of this verification's own are added --

  * ``mine``       -- a different lambda / w0 / f near-focus ladder
  * ``b4``         -- the ONE suite leg outside the tau fixtures that the rule
                      moves, reproduced from ``test_audit2609_b4_collins_
                      transport.py``'s ``mismatch_matrix`` fixture, and scored
                      against a NON-PARAXIAL oracle so the question "which
                      kernel is right there" can be answered rather than
                      asserted
  * ``wide_far``   -- a wide-envelope leg of this verification's own, built to
                      trip the rule on ANGLE, scored against the same oracle.

The non-paraxial oracle is a closed-form angular spectrum: the input field is
``exp(-a r^2)`` with COMPLEX ``a`` (Gaussian amplitude times the parabolic
carrier), whose 2-D Fourier transform is ``(pi/a) exp(-pi^2 f^2 / a)``
analytically, so nothing has to be sampled on the aliased input lattice.  That
spectrum is propagated with the EXACT transfer function
``exp(i k z sqrt(1 - lambda^2 f^2))`` and inverted by a 1-D Hankel quadrature
onto the readout's own abscissae.  It shares no machinery with the library.

Usage (BLAS pinned on the COMMAND LINE)::

    PYTHONPATH=<tree> python validation/probe_verify_c5/v_item1_kernel.py OUT.json
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                            # noqa: E402
from scipy.special import j0                                  # noqa: E402

from _vd import dig, write                                    # noqa: E402

from lumenairy.propagators import carrier as CA               # noqa: E402
from lumenairy.propagators.carrier import _collins_transport  # noqa: E402


# ---------------------------------------------------------------- helpers --
def ax(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def gauss(n, dx, w):
    g = ax(n, dx)
    return np.exp(-(g[None, :] ** 2 + g[:, None] ** 2)
                  / float(w) ** 2).astype(np.complex128)


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def rel_pf(a, b):
    """Piston-free relative L2 -- the b4 file's own score."""
    a, b = np.asarray(a), np.asarray(b)
    ov = np.vdot(b, a)
    return rel(a / (ov / abs(ov)) if abs(ov) > 0 else a, b)


def q_field(n, dxo, q_in, z, lam):
    """The PARAXIAL analytic Gaussian, whole-function q form, piston in."""
    k = 2.0 * np.pi / lam
    xo = ax(n, dxo)
    r2 = xo[None, :] ** 2 + xo[:, None] ** 2
    return (np.exp(1j * k * z) / (1.0 + z / q_in)
            * np.exp(1j * k * r2 / (2.0 * (q_in + z))))


def theta_env_of(env, dx, lam):
    S = np.fft.fft2(np.ascontiguousarray(env, dtype=np.complex128))
    return max(CA._collins_envelope_half_angle(S, dx, dx, lam))


# ------------------------------------------------- the non-paraxial oracle --
def exact_gaussian_field(w, R0, lam, z, n_out, dx_out, n_f=None, f_cap=None):
    """EXACT scalar (Helmholtz) field of ``exp(-r^2/w^2) exp(i k r^2 / 2 R0)``
    a distance ``z`` on, on the readout's own ``(n_out, dx_out)`` lattice.

    Analytic spectrum, exact transfer function, 1-D Hankel quadrature over the
    grid's UNIQUE radii.  Returns ``(field, meta)``.
    """
    k = 2.0 * np.pi / float(lam)
    a = 1.0 / float(w) ** 2 - 1j * k / (2.0 * float(R0))
    inv_a = 1.0 / a
    # |exp(-pi^2 f^2/a)| = exp(-pi^2 f^2 Re(1/a)); cut at e^-46 (~1e-20).
    f_decay = float(np.sqrt(46.0 / (np.pi ** 2 * inv_a.real)))
    F = float(f_cap) if f_cap else min(f_decay, 0.90 / float(lam))
    xo = ax(n_out, dx_out)
    r2 = xo[None, :] ** 2 + xo[:, None] ** 2
    rr = np.round(np.sqrt(r2), 15)
    rho = np.unique(rr)
    if n_f is None:
        ph_H = abs(k * z) * (float(lam) * F) ** 2 / 2.0
        ph_J = 2.0 * np.pi * F * float(rho.max())
        n_f = int(max(2.0e5, 48.0 * (ph_H + ph_J) / (2.0 * np.pi)))
        n_f = min(n_f, 4_000_001)
    f = np.linspace(0.0, F, int(n_f))
    lf2 = (float(lam) * f) ** 2
    H = np.exp(1j * k * float(z) * np.sqrt(np.maximum(1.0 - lf2, 0.0)))
    w_f = (np.pi * inv_a) * np.exp(-(np.pi ** 2) * inv_a * f * f) * H * f
    vals = np.empty(rho.size, dtype=np.complex128)
    step = max(1, int(4.0e7 // max(1, f.size)))
    for i0 in range(0, rho.size, step):
        blk = rho[i0:i0 + step]
        B = j0(2.0 * np.pi * np.outer(blk, f)) * w_f[None, :]
        vals[i0:i0 + step] = 2.0 * np.pi * np.trapezoid(B, f, axis=-1)
    look = {v: c for v, c in zip(rho.tolist(), vals.tolist())}
    out = np.array([look[v] for v in rr.ravel().tolist()],
                   dtype=np.complex128).reshape(r2.shape)
    return out, dict(F=F, n_f=int(n_f), n_rho=int(rho.size),
                     f_decay=f_decay, lam_F=float(lam) * F)


# ============================================================== fixture A ===
H2 = dict(w0=15.915e-6, theta=20.0e-3, f=20.0e-3, n=512, dx=8e-6)
H2['lam'] = float(np.pi * H2['w0'] * H2['theta'])
H2['zr'] = float(np.pi * H2['w0'] ** 2 / H2['lam'])


def h2_env():
    q = complex(-H2['f'], -H2['zr'])
    w_in = float(np.sqrt(H2['lam'] / (np.pi * np.imag(1.0 / q))))
    R_in = 1.0 / float(np.real(1.0 / q))
    return R_in, w_in, gauss(H2['n'], H2['dx'], w_in)


def h2_leg(d, gk, st=None, from_focus=False):
    R_in, _w, env = h2_env()
    z0 = (-R_in) if from_focus else H2['f']
    return _collins_transport(
        env, R_in, z0 - d, H2['lam'], H2['dx'], H2['dx'],
        dx_out=H2['dx'], dy_out=H2['dx'], N_out_x=H2['n'], N_out_y=H2['n'],
        R_ref=float('inf'), gap_kernel=gk, on_collins_sampling='ignore',
        stats_out=st)


# ============================================================== fixture B ===
F3 = dict(lam=1.064e-6, n=1024, dx=4e-6, w=0.30e-3, R=-40e-3,
          dx_out=5.6447e-06, n_out=128)


def f3_leg(dz, gk, st=None):
    return _collins_transport(
        gauss(F3['n'], F3['dx'], F3['w']), F3['R'], -F3['R'] - dz,
        F3['lam'], F3['dx'], F3['dx'], dx_out=F3['dx_out'],
        dy_out=F3['dx_out'], N_out_x=F3['n_out'], N_out_y=F3['n_out'],
        R_ref=float('inf'), gap_kernel=gk, on_collins_sampling='ignore',
        stats_out=st)


def f3_oracle(dz):
    q_in = 1.0 / (1.0 / F3['R'] + 1j * F3['lam'] / (np.pi * F3['w'] ** 2))
    return q_field(F3['n_out'], F3['dx_out'], q_in, -F3['R'] - dz, F3['lam'])


# ============================================================== fixture C ===
MY = dict(lam=0.633e-6, n=1024, dx=3e-6, w=0.20e-3, R=-30e-3,
          dx_out=3.1e-06, n_out=128)


def my_leg(dz, gk, st=None):
    return _collins_transport(
        gauss(MY['n'], MY['dx'], MY['w']), MY['R'], -MY['R'] - dz,
        MY['lam'], MY['dx'], MY['dx'], dx_out=MY['dx_out'],
        dy_out=MY['dx_out'], N_out_x=MY['n_out'], N_out_y=MY['n_out'],
        R_ref=float('inf'), gap_kernel=gk, on_collins_sampling='ignore',
        stats_out=st)


def my_oracle(dz):
    q_in = 1.0 / (1.0 / MY['R'] + 1j * MY['lam'] / (np.pi * MY['w'] ** 2))
    return q_field(MY['n_out'], MY['dx_out'], q_in, -MY['R'] - dz, MY['lam'])


# =========================================== fixture D -- mismatched carrier =
def mismatch_env(cfg, fr):
    lam, n, dx = cfg['lam'], cfg['n'], cfg['dx']
    k = 2.0 * np.pi / lam
    g = ax(n, dx)
    r2 = g[None, :] ** 2 + g[:, None] ** 2
    R0 = cfg['R0']
    return (np.exp(-r2 / cfg['w'] ** 2)
            * np.exp(1j * k * r2 / (2.0 * R0))
            * np.exp(-1j * k * r2 / (2.0 * fr * R0))).astype(np.complex128)


def mismatch_leg(cfg, fr, gk, st=None):
    """ONE physical field ``exp(-r^2/w^2) exp(i k r^2 / 2 R0)`` re-enveloped
    against a MISMATCHED carrier ``R = fr*R0``: the envelope then carries a
    residual lens, which is what makes ``theta_env`` large far from a focus."""
    env = mismatch_env(cfg, fr)
    E = _collins_transport(
        env, fr * cfg['R0'], -cfg['R0'], cfg['lam'], cfg['dx'], cfg['dx'],
        dx_out=cfg['dxo'], dy_out=cfg['dxo'], N_out_x=cfg['nout'],
        N_out_y=cfg['nout'], R_ref=float('inf'), gap_kernel=gk,
        on_collins_sampling='ignore', stats_out=st)
    return E, env


def mismatch_oracles(cfg):
    lam, R0, z = cfg['lam'], cfg['R0'], -cfg['R0']
    q_in = 1.0 / (1.0 / R0 + 1j * lam / (np.pi * cfg['w'] ** 2))
    par = q_field(cfg['nout'], cfg['dxo'], q_in, z, lam)
    ex, meta = exact_gaussian_field(cfg['w'], R0, lam, z,
                                    cfg['nout'], cfg['dxo'])
    return par, ex, meta


B4 = dict(lam=1.31e-6, n=1024, w=1.0e-3, R0=-1.0e-3 / 0.05)
B4['dx'] = 2.0 * 4.0 * B4['w'] / B4['n']
B4['w0'] = B4['lam'] * abs(B4['R0']) / (np.pi * B4['w'])
B4['dxo'] = B4['w0'] / 8.0
B4['nout'] = 64

WF = dict(lam=1.55e-6, n=1024, w=0.80e-3, R0=-0.80e-3 / 0.04)
WF['dx'] = 2.0 * 4.0 * WF['w'] / WF['n']
WF['w0'] = WF['lam'] * abs(WF['R0']) / (np.pi * WF['w'])
WF['dxo'] = WF['w0'] / 8.0
WF['nout'] = 64


# ------------------------------------------------------------------- band --
def bisect_edge(leg, theta_env, lam, tau, lo, hi, iters=80):
    """The distance-to-focus at which the PREDICTED departure crosses tau,
    found on the running build by bisection on the leg itself."""
    def dep_at(d):
        st = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            leg(d, 'auto', st)
        z_eff = abs(float(st['abcd'][1]) / float(st['abcd'][0]))
        return CA._collins_exact_kernel_departure(z_eff, theta_env, lam)
    if not (dep_at(lo) > tau > dep_at(hi)):
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if dep_at(mid) > tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main(out_path):
    tau = CA._GAP_KERNEL_ACCURACY_TAU
    res = {'tau': tau, 'digests': {}}
    D = res['digests']
    warnings.simplefilter('ignore')

    # ---- fixture A: the hygiene-2 ladder ---------------------------------
    rows = []
    for d in (1e-6, 3e-6, 10e-6, 30e-6, 100e-6, 300e-6, 1e-3, 3e-3, 5e-3):
        st = {}
        E = h2_leg(d, 'auto', st)
        rows.append(dict(d=d, z_eff=float(st['abcd'][1] / st['abcd'][0]),
                         k4=st['k4'], kernel=st['kernel'],
                         departure=st.get('kernel_departure'),
                         has_dep_key=('kernel_departure' in st)))
        D['h2/auto/%g' % d] = dig(E)
    res['h2_ladder'] = rows
    _R, _w, _env = h2_env()
    res['h2_theta_env'] = theta_env_of(_env, H2['dx'], H2['lam'])
    res['h2_worst_departure'] = max((r['departure'] or 0.0) for r in rows)

    # ---- fixture B: F3 ---------------------------------------------------
    rows = []
    for dz in (1e-6, 10e-6, 100e-6, 1e-3, 5e-3):
        st = {}
        Ea = f3_leg(dz, 'auto', st)
        Ef = f3_leg(dz, 'fresnel')
        Ex = None
        try:
            Ex = f3_leg(dz, 'exact')
            ex_err = rel(Ex, f3_oracle(dz))
        except ValueError as exc:
            ex_err = None
            res.setdefault('f3_exact_refused', {})['%g' % dz] = str(exc)[:160]
        T = f3_oracle(dz)
        rows.append(dict(dz=dz, z_eff=float(st['abcd'][1] / st['abcd'][0]),
                         k4=st['k4'], kernel=st['kernel'],
                         departure=st.get('kernel_departure'),
                         auto_vs_oracle=rel(Ea, T),
                         fresnel_vs_oracle=rel(Ef, T),
                         exact_vs_oracle=ex_err))
        D['f3/auto/%g' % dz] = dig(Ea)
        D['f3/fresnel/%g' % dz] = dig(Ef)
        if Ex is not None:
            D['f3/exact/%g' % dz] = dig(Ex)
    res['f3_ladder'] = rows
    res['f3_theta_env'] = theta_env_of(
        gauss(F3['n'], F3['dx'], F3['w']), F3['dx'], F3['lam'])

    # ---- fixture C: mine --------------------------------------------------
    rows = []
    for dz in (0.5e-6, 2e-6, 20e-6, 200e-6, 2e-3):
        st = {}
        Ea = my_leg(dz, 'auto', st)
        Ef = my_leg(dz, 'fresnel')
        T = my_oracle(dz)
        rows.append(dict(dz=dz, z_eff=float(st['abcd'][1] / st['abcd'][0]),
                         k4=st['k4'], kernel=st['kernel'],
                         departure=st.get('kernel_departure'),
                         auto_vs_oracle=rel(Ea, T),
                         fresnel_vs_oracle=rel(Ef, T)))
        D['mine/auto/%g' % dz] = dig(Ea)
        D['mine/fresnel/%g' % dz] = dig(Ef)
    res['mine_ladder'] = rows
    res['mine_theta_env'] = theta_env_of(
        gauss(MY['n'], MY['dx'], MY['w']), MY['dx'], MY['lam'])

    # ---- the band, bisected ----------------------------------------------
    band = {}
    if tau is not None:
        for tag, leg, th, lam in (
                ('f3', f3_leg, res['f3_theta_env'], F3['lam']),
                ('h2_from_A0',
                 lambda d, gk, st=None: h2_leg(d, gk, st, from_focus=True),
                 res['h2_theta_env'], H2['lam']),
                ('mine', my_leg, res['mine_theta_env'], MY['lam'])):
            k = 2.0 * np.pi / lam
            band[tag] = dict(
                theta_env=th,
                z_eff_threshold=(8.0 * tau
                                 / (CA._QUARTIC_RMS_MOMENT * k * th ** 4)),
                bisected=bisect_edge(leg, th, lam, tau, 1e-9, 1e-3))
    res['band'] = band

    # ---- the mismatched-carrier legs, against BOTH oracles ---------------
    for tag, cfg, frs in (('b4', B4, (1.00, 0.99, 0.98, 0.95, 0.90)),
                          ('wide_far', WF, (1.00, 0.95, 0.90, 0.85))):
        par, exo, meta = mismatch_oracles(cfg)
        exo2, meta2 = exact_gaussian_field(
            cfg['w'], cfg['R0'], cfg['lam'], -cfg['R0'], cfg['nout'],
            cfg['dxo'], n_f=meta['n_f'] // 2 + 1)
        rows = []
        for fr in frs:
            st = {}
            Ea, env = mismatch_leg(cfg, fr, 'auto', st)
            Ef, _ = mismatch_leg(cfg, fr, 'fresnel')
            try:
                Ex, _ = mismatch_leg(cfg, fr, 'exact')
            except ValueError:
                Ex = None
            A = float(st['abcd'][0])
            rows.append(dict(
                fr=fr, A=A, B=float(st['abcd'][1]),
                z_eff=(float(st['abcd'][1]) / A) if A else None,
                k4=st['k4'], kernel=st['kernel'],
                departure=st.get('kernel_departure'),
                theta_env=theta_env_of(env, cfg['dx'], cfg['lam']),
                auto_vs_paraxial=rel_pf(Ea, par),
                fresnel_vs_paraxial=rel_pf(Ef, par),
                exact_vs_paraxial=(rel_pf(Ex, par) if Ex is not None else None),
                auto_vs_exact_oracle=rel_pf(Ea, exo),
                fresnel_vs_exact_oracle=rel_pf(Ef, exo),
                exact_vs_exact_oracle=(rel_pf(Ex, exo)
                                       if Ex is not None else None),
                exact_minus_fresnel=(rel(Ex, Ef) if Ex is not None else None),
            ))
            D['%s/auto/%g' % (tag, fr)] = dig(Ea)
            D['%s/fresnel/%g' % (tag, fr)] = dig(Ef)
        res[tag + '_rows'] = rows
        res[tag + '_oracle'] = dict(
            meta=meta, meta_half=meta2,
            quadrature_selfcheck=rel(exo2, exo),
            paraxial_vs_exact_oracle=rel_pf(par, exo),
            cfg=dict(cfg))

    # ---- the ORACLE's own validation -------------------------------------
    # Two-sided, and independent of anything the library computes: the
    # non-paraxial departure of a converging Gaussian is
    # ``sqrt(3/2) k |z| theta^4 / 8`` to leading order, so sweeping NA by 10x
    # must move the reading by 1e4 with a FIXED ratio to the law.  A quadrature
    # error would not scale that way, and an oracle that silently reproduced
    # the paraxial answer would read zero.
    vrows = []
    for na in (0.005, 0.01, 0.02, 0.05):
        cfg = dict(lam=1.31e-6, w=1.0e-3)
        R0 = -cfg['w'] / na
        z = -R0
        w0 = cfg['lam'] * abs(R0) / (np.pi * cfg['w'])
        dxo, nout = w0 / 8.0, 64
        q_in = 1.0 / (1.0 / R0 + 1j * cfg['lam'] / (np.pi * cfg['w'] ** 2))
        par = q_field(nout, dxo, q_in, z, cfg['lam'])
        exo, m = exact_gaussian_field(cfg['w'], R0, cfg['lam'], z, nout, dxo)
        exo_h, _ = exact_gaussian_field(cfg['w'], R0, cfg['lam'], z, nout,
                                        dxo, n_f=m['n_f'] // 2 + 1)
        k = 2.0 * np.pi / cfg['lam']
        law = CA._QUARTIC_RMS_MOMENT * k * abs(z) * na ** 4 / 8.0
        got = rel_pf(par, exo)
        vrows.append(dict(na=na, z=z, law=law, measured=got,
                          ratio=got / law, quad_selfcheck=rel(exo_h, exo)))
    res['oracle_validation'] = vrows

    # ---- the opt-out ------------------------------------------------------
    CA._GAP_KERNEL_ACCURACY_TAU = None
    try:
        opt = {}
        keys = {}
        for d in (1e-6, 10e-6, 100e-6):
            st = {}
            opt['h2/%g' % d] = dig(h2_leg(d, 'auto', st))
            keys['h2/%g' % d] = dict(kernel=st['kernel'],
                                     has_dep=('kernel_departure' in st))
        for dz in (1e-6, 10e-6, 100e-6, 1e-3, 5e-3):
            st = {}
            opt['f3/%g' % dz] = dig(f3_leg(dz, 'auto', st))
            keys['f3/%g' % dz] = dict(kernel=st['kernel'],
                                      has_dep=('kernel_departure' in st))
        for fr in (0.95, 0.90):
            st = {}
            Eo, _ = mismatch_leg(B4, fr, 'auto', st)
            opt['b4/%g' % fr] = dig(Eo)
            keys['b4/%g' % fr] = dict(kernel=st['kernel'],
                                      has_dep=('kernel_departure' in st))
        res['optout_digests'] = opt
        res['optout_keys'] = keys
    finally:
        CA._GAP_KERNEL_ACCURACY_TAU = tau

    write(out_path, res)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1
         else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'v_item1_head_win.json'))

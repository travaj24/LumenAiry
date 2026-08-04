# SAMPLING ADEQUACY for every arm this study quotes, stated the way the
# campaign requires it: the AMPLITUDE-WEIGHTED wrapped nearest-neighbour phase
# step at p99.9 against pi -- never a max (DIAG_LAST_GROUP_DECENTRE artefact 4:
# one skirt pixel sets a max, and ``_phase_gradient`` returns exactly that).
#
# Two quantities per arm:
#
#   ENVELOPE step -- of the object whose GRADIENT becomes the launch direction.
#       If this is not << pi the launch directions are meaningless.  This is
#       the statistic that separates the shipped readout (saturated at pi) from
#       the exact-eikonal one.
#   INTEGRAND step -- of the Rayleigh-Sommerfeld integrand at the farthest
#       image point scored, WRAPPED (the unwrapped form ``rs_spot`` prints is
#       contaminated by the 2-D unwrap's own jumps, which the RS kernel cannot
#       see because it only consumes exp(i*ph0)).
#
# usage:  ORDERS='0,0 -4,-2' python fc_sampling_121.py
import os
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM


def env_step_stats(env):
    """Amplitude-weighted wrapped nearest-neighbour phase step of ``env``:
    ``(p50, p99.9, max)`` in radians.  The weight is the smaller of the two
    neighbours' |E| over the peak, so a step across two dark pixels cannot
    dominate -- the same weighting ``rs_spot`` applies to its own integrand
    statistic."""
    ph = np.angle(env)
    a = np.abs(env)
    mx = max(float(a.max()), 1e-300)

    def _w(x):
        return (x + np.pi) % (2 * np.pi) - np.pi

    st = np.concatenate([
        (np.abs(_w(ph[:, 1:] - ph[:, :-1]))
         * np.minimum(a[:, 1:], a[:, :-1]) / mx).ravel(),
        (np.abs(_w(ph[1:, :] - ph[:-1, :]))
         * np.minimum(a[1:], a[:-1]) / mx).ravel()])
    st = st[np.isfinite(st)]
    if not st.size:
        return np.nan, np.nan, np.nan
    return (float(np.percentile(st, 50)), float(np.percentile(st, 99.9)),
            float(st.max()))


def main():
    orders = os.environ.get('ORDERS', '0,0 -4,-2').split()
    rn, rs = int(os.environ.get('RN', '1024')), int(os.environ.get('RS', '4'))
    print(FI._provenance())
    print(f"pi = {np.pi:.4f}\n")
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    print(f"{'order':>8} {'arm':>34} {'env p50':>9} {'env p99.9':>10} "
          f"{'env max':>9} {'RSp99.9 cyc':>12}")
    for o in orders:
        m, n = (int(v) for v in o.split(','))
        L, M = m * LAM / period, n * LAM / period
        ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L]), M=np.array([M]),
                             N=np.array([np.sqrt(1 - L * L - M * M)]),
                             wavelength=LAM, alive=np.ones(1, bool),
                             opd=np.zeros(1)),
                   C.post_surfaces(post), LAM,
                   output_filter='last').image_rays
        xci, yci = float(ch.x[0]), float(ch.y[0])
        # the oracle arm: the object differentiated is env_doe itself
        e50, e999, emx = env_step_stats(env)
        la_ = FI.oracle_launch(env, R, dx, L, M, 321, 3.0, 1, post, 5e-3)
        _, w999, _ = FI._wrapped_step(*la_[:7], 5e-3, xci, yci, la_[7], 61,
                                      0.4e-6)
        print(f"{o:>8} {'oracle (env_doe residual)':>34} {e50:9.5f} "
              f"{e999:10.5f} {emx:9.5f} {w999:12.5f}")
        for taper in ('on', 'off'):
            res, _w = FI.run_chain(post, env, R, dx, L, M, rs, taper, 0)
            for split in ('parabola', 'exact'):
                lb = FI.chain_launch(res, L, M, 9999, 3.0, 1, post, 5e-3,
                                     split)
                nl = lb[7]
                # ``_residual_of`` rebuilds the object ``chain_launch``
                # finite-differences (the envelope AFTER the analytic
                # reference for this split has been divided out), on the same
                # crop, so the statistic is of the differentiated quantity
                # itself rather than of the total launch phase.
                e50, e999, emx = env_step_stats(
                    _residual_of(res, L, M, split))
                _, w999, _ = FI._wrapped_step(*lb[:7], 5e-3, xci, yci, nl, 61,
                                              0.4e-6)
                print(f"{o:>8} {f'chain taper={taper} split={split}':>34} "
                      f"{e50:9.5f} {e999:10.5f} {emx:9.5f} {w999:12.5f}")
    return 0


def _residual_of(res, L, M, split):
    """The object ``chain_launch`` finite-differences, on the full crop."""
    import lumenairy.propagators.carrier as CM
    K0 = 2 * np.pi / LAM
    fld, Rk, dxk = np.asarray(res.field), float(res.R), float(res.dx)
    st = [s for s in res.stages if not s.get('target')][-1]
    Lk, Mk = st.get('L_out', L), st.get('M_out', M)
    envk = CM.carrier_referenced_envelope(fld, Rk, LAM, dxk)
    nn = envk.shape[0]
    u = (np.arange(nn) - nn / 2) * dxk
    envk = envk * np.exp(-1j * K0 * (Lk * u[None, :] + Mk * u[:, None]))
    if split == 'exact':
        r2 = u[None, :] ** 2 + u[:, None] ** 2
        npar = np.sqrt(max(1.0 - Lk * Lk - Mk * Mk, 0.0))
        uu = u[None, :] + Rk * Lk / npar
        vv = u[:, None] + Rk * Mk / npar
        Wref = np.sign(Rk) * (np.sqrt(uu * uu + vv * vv + Rk * Rk)
                              - abs(Rk) / npar)
        Wref = Wref - (Lk * u[None, :] + Mk * u[:, None])
        envk = envk * np.exp(-1j * K0 * (Wref - r2 / (2.0 * Rk)))
    wk = CM._envelope_amp_radius(envk, dxk, dxk)
    half = int(np.ceil(3.0 * wk / dxk))
    c = nn // 2
    return envk[max(c - half, 0):min(c + half + 1, nn),
                max(c - half, 0):min(c + half + 1, nn)]


if __name__ == '__main__':
    sys.exit(main())

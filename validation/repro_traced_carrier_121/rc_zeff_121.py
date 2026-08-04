# D121 RESIDUAL CLOSURE -- how sensitive is the chain to the ENVELOPE leg's
# effective distance?
#
# ``_tilt_obliquity``'s docstring records an unimplemented correction: under a
# carrier tilt the residual envelope's own diffraction wants an effective
# distance ``z/(1-L^2-M^2)^{3/2}`` ALONG the tilt and ``z/(1-L^2-M^2)^{1/2}``
# ACROSS it, while the chain runs every leg at the bare ``z``.  At design
# 121's largest order that is +0.32 % / +0.11 %; at the SMALLEST it is
# +0.02 % / +0.007 %.
#
# This prices that axis without implementing it, by scaling the Sziklas-
# Siegman reduced distance ``z_eff`` in ``_carrier_step_fast`` by a constant
# ``s`` and leaving EVERYTHING geometric (``R_out``, ``dx_out``, the piston,
# the 1/m amplitude) computed from the true ``z``.  ``s = 1`` must reproduce
# the shipped run BIT FOR BIT -- asserted as the null.
#
# If EE3 is flat across ``s`` at the +-0.3 % level, the obliquity correction
# cannot be worth a tenth of a point and the axis is closed by measurement
# rather than by argument.
#
# SCRIPT-SIDE ONLY -- ``carrier.py`` is not edited by this probe.
#
# usage:  ORDERS='-1,0 -4,0' SCALES='0.99,0.997,1,1.003,1.01' python rc_zeff_121.py
import hashlib
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as C                                       # noqa: E402
import fc_instrument_121 as FI                                 # noqa: E402
import hybrid_localize_121 as H                                # noqa: E402
import rc_readout_121 as RD                                    # noqa: E402
import lumenairy.propagators.carrier as _CM                    # noqa: E402
from approx_common import Patch                                # noqa: E402
from lumenairy.raytrace import RayBundle, trace                # noqa: E402

LAM = C.LAM
BACK = 5.0e-3
_S = [1.0]


def _step_scaled(E_env, R, z, wavelength, dx, dy):
    """``_carrier_step_fast`` with the ENVELOPE leg's reduced distance scaled
    by ``_S[0]`` and nothing else touched."""
    R_out = R + z
    m = R_out / R
    z_eff = z * R / R_out * _S[0]
    u_out = _CM.fresnel_tf_propagate(E_env, z_eff, wavelength, dx, dy)
    k = 2.0 * np.pi / wavelength
    piston = np.exp(1j * k * (z * z / R_out))
    env_out = (piston / m) * u_out
    if np.iscomplexobj(E_env) and env_out.dtype != E_env.dtype:
        env_out = env_out.astype(E_env.dtype)
    return _CM.CarrierReferencedField(env_out, R_out, m * dx)


def main():
    orders = os.environ.get('ORDERS', '-1,0 -4,0').split()
    scales = [float(v) for v in os.environ.get(
        'SCALES', '0.99,0.997,1,1.003,1.01').split(',')]
    rn, rs, clip = 1024, 4, 3.0
    print(FI._provenance(), flush=True)
    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=rn, rs=rs)
    for o in orders:
        m_, n_ = (int(v) for v in o.split(','))
        L, M = m_ * LAM / period, n_ * LAM / period
        ob = 1.0 / np.sqrt(1.0 - L * L - M * M)
        ch = trace(RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                             L=np.array([L]), M=np.array([M]),
                             N=np.array([np.sqrt(1 - L * L - M * M)]),
                             wavelength=LAM, alive=np.ones(1, bool),
                             opd=np.zeros(1)),
                   C.post_surfaces(post), LAM,
                   output_filter='last').image_rays
        xci, yci = float(ch.x[0]), float(ch.y[0])
        print(f"\n########## ORDER ({m_:+d},{n_:+d}) -- the obliquity "
              f"correction this order WANTS is s = ob^3 = "
              f"{ob ** 3:.6f} (along) / ob = {ob:.6f} (across) ##########",
              flush=True)
        print(f"  {'s':>9} {'EE3 area':>9} {'d vs s=1':>9} {'EE6 area':>9} "
              f"{'FWHM um':>8} {'sha':>18}", flush=True)
        base = None
        for s in scales:
            t0 = time.time()
            _S[0] = s
            with Patch([(_CM, '_carrier_step_fast', _step_scaled)]):
                res, _w = FI.run_chain(post, env, R, dx, L, M, rs, 'off', 0)
                a = FI.chain_launch(res, L, M, 9999, clip, 1, post, BACK,
                                    'exact')
            r = H.rs_spot(*a[:7], BACK, xci, yci, dx_out=0.4e-6, n_out=61,
                          nl=a[7])
            I = np.ascontiguousarray(r['I'])
            cx, cy = RD.centroid(I, r['ax'])
            e3 = RD.ee(I, r['ax'], cx, cy, 3e-6, 'area') * 100
            e6 = RD.ee(I, r['ax'], cx, cy, 6e-6, 'area') * 100
            dig = hashlib.sha256(I.tobytes()).hexdigest()[:16]
            if s == 1.0:
                base = e3
            print(f"  {s:>9.5f} {e3:9.4f} "
                  f"{('' if base is None else f'{e3 - base:+9.4f}')} "
                  f"{e6:9.4f} {r['fwhm'] * 1e6:8.3f} {dig:>18}   "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        _S[0] = 1.0
    print("\nNULL: the s = 1 row must be byte-identical to the shipped chain "
          "(sha in rc_readout's table).", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

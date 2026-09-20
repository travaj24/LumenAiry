"""Where does the `corr > 1 - 1e-6` bar's 3.17e-07 come from?

The merit is P(a) = sum|L a|^2 = a^T (L^H L) a with L the Collins leg, so
grad P = 2 (L^H L) a.  "grad is proportional to a" holds EXACTLY iff a is an
eigenvector of L^H L -- i.e. iff the leg is a scaled isometry on the span of
the fixture.  The correlation's shortfall is therefore the leg's own departure
from a scaled isometry, restricted to the support mask, and that is a quantity
the running build can MEASURE.  This probe measures it and checks the
identity 1 - corr ~ (1/2) * r^2 with r the relative residual of grad/2 against
its best multiple of a.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
vlib.anchor(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'lumenairy'))
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import lumenairy.propagators.carrier as CA

WL, N, DX, R_IN, Z, R_REF = 633e-9, 64, 8e-6, -0.05, 5e-3, -0.045
ax = (np.arange(N) - N // 2) * DX
X, Y = np.meshgrid(ax, ax)
env = np.exp(-(X**2 + Y**2) / (60e-6)**2).astype(np.complex128)


def merit(amp):
    e = amp.astype(jnp.complex128)
    out = CA._collins_transport(e, R_IN, Z, WL, DX, DX, dx_out=DX, dy_out=DX,
                                N_out_x=N, N_out_y=N, R_ref=R_REF,
                                gap_kernel='fresnel',
                                on_collins_sampling='ignore')
    return jnp.sum(jnp.abs(out)**2)


a = np.real(env)
g = np.asarray(jax.grad(merit)(jnp.asarray(a)))
m = a > 0.05 * a.max()
corr = float(np.corrcoef(g[m], a[m])[0, 1])

gm, am = g[m], a[m]
c = float(np.dot(gm, am) / np.dot(am, am))
r = float(np.linalg.norm(gm - c * am) / np.linalg.norm(c * am))

# the leg's own departure from a scaled isometry, measured directly:
# P(a)/||a||^2 evaluated on several unit inputs must be constant for an
# isometry.  Sample with the fixture and with two orthogonal perturbations.
rng = np.random.default_rng(7)
ratios = []
for _ in range(6):
    v = rng.standard_normal(a.shape) * m
    v /= np.linalg.norm(v)
    ratios.append(float(merit(jnp.asarray(v))))
iso_spread = float((max(ratios) - min(ratios)) / np.mean(ratios))

out = dict(build=vlib.build_tag(), corr=corr, one_minus_corr=1.0 - corr,
           residual_rel_r=r, half_r_squared=0.5 * r * r,
           ratio_check=(1.0 - corr) / (0.5 * r * r),
           isometry_spread_over_mask=iso_spread,
           power_in=float(np.sum(np.abs(a)**2)),
           power_out=float(merit(jnp.asarray(a))),
           power_ratio=float(merit(jnp.asarray(a)) / np.sum(np.abs(a)**2)))
for k, v in out.items():
    print(f"{k:34s} {v}")
vlib.write_json(out, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                f"v0_corr_derivation_{vlib.build_tag().split('-')[0].lower()}.json"))


# ---------------------------------------------------------------------------
# How much does 1 - corr MOVE under a legitimate change of fixture?  If a
# modest change in the envelope width crosses the test's pinned 1e-6, the bar
# is measuring the fixture and not the port.
if os.environ.get('VHYG2_SWEEP'):
    sweep = {}
    for w in (30e-6, 40e-6, 50e-6, 60e-6, 70e-6, 80e-6, 100e-6):
        e = np.exp(-(X**2 + Y**2) / w**2).astype(np.complex128)
        aa = np.real(e)
        gg = np.asarray(jax.grad(merit)(jnp.asarray(aa)))
        mm = aa > 0.05 * aa.max()
        cc = float(np.corrcoef(gg[mm], aa[mm])[0, 1])
        sweep[f"{w:.1e}"] = 1.0 - cc
        print(f"w={w:.1e}  1-corr={1.0 - cc:.6e}  "
              f"{'OVER the pinned 1e-6' if (1.0 - cc) > 1e-6 else 'under'}")
    vlib.write_json({'build': vlib.build_tag(), 'one_minus_corr_vs_w': sweep},
                    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 f"v0_corr_width_sweep_"
                                 f"{vlib.build_tag().split('-')[0].lower()}.json"))

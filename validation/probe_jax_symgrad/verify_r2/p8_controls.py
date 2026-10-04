"""Claim (7): two NOT-routed controls at their symmetric points, own
fixtures, on HEAD and on the base tree (LUMROOT): the 1-D RCWA d / d(angle)
at exactly normal incidence, and BORStack d / d(n_ring) at a radially
homogeneous layer.  AD (jit(jacrev)) vs a premise-checked FD; values saved
for a HEAD-vs-base comparison.  python p8_controls.py <tag>"""
import json
import sys
import warnings

from _vc import BUILD, fd_rich, jax, jnp, np, premise_ok, rel

from lumenairy.elements.bor.bor_stack import BORStack
from lumenairy.elements.rcwa import rcwa_efficiency_1d

out = {"build": BUILD}


def r1d(pol):
    def f(t, xp):
        _o, R, T = rcwa_efficiency_1d(0.9, xp.asarray(2.6 + 0.05j), 1.2, 1.5,
                                      1.0, 0.35, 0.45, 1.0, angle=t,
                                      polarization=pol, n_orders=13)
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    return f


def bor(m):
    def f(nr, xp):
        s = BORStack(2.5, m, N=48, n_superstrate=1.3, n_substrate=1.6)
        s.add_layer(0.4, rings=(0.7, 0.4, nr, 2.2))
        s.set_source(wavelength=2 * np.pi / 2.4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = s.solve()
        R, T = xp.asarray(r["R"]), xp.asarray(r["T"])
        return xp.concatenate([xp.ravel(R), xp.ravel(T)])
    return f


for name, f, x0, cast in (("rcwa1d_te", r1d("te"), 0.0, float),
                          ("rcwa1d_tm", r1d("tm"), 0.0, float),
                          ("bor_m1", bor(1), 2.2, float),
                          ("bor_m2", bor(2), 2.2, float)):
    fd, ratio, ex = fd_rich(lambda t: f(cast(t), np) if "rcwa" in name
                            else np.asarray(f(jnp.asarray(t), jnp)), x0)
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(
        jnp.asarray(x0)))
    v = np.asarray(jax.jit(lambda t: f(t, jnp))(jnp.asarray(x0)))
    out[name] = dict(err=rel(g, fd), premise=premise_ok(ratio, ex),
                     g=g.tolist(), v=v.tolist())
    print(name, out[name]["err"], out[name]["premise"], flush=True)
json.dump(out, open(f"p8_controls_{sys.argv[1]}_{BUILD}.json", "w"))

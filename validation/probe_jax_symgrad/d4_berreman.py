"""D4: the Berreman 4x4 JAX twin (``berreman_jones_1d`` on jnp inputs).

Stack: n_sup 1.45 | layer 1: eps = 2.56 I + delta X, t 0.3 um |
layer 2: eps 2.1, t 0.1 um | n_sub 1.0, wl 1 um, normal incidence.

A TRACED (3,3) tensor routes to ``_offplane_solve_jax`` (_berreman_jax.py
~414, eig site 3: the layer Delta of ``_layer_M_gen_jax``); concrete tensors
with a traced angle route to ``_solve_jax`` (~94, eig site 1: sup, sub and
every layer Delta).  At normal incidence an isotropic layer's Delta has two
exactly degenerate pairs (+-i n, x2).

Parameters:
  exy   X = [[0,1,0],[1,0,0],[0,0,0]]   splits the pairs (symmetry-breaking)
  xxyy  X = diag(1,-1,0)                splits the pairs (symmetry-breaking)
  iso   X = I                           keeps them        (control)
  angle / phi with CONCRETE closed-over tensors (``_solve_jax``):
        angle_iso   isotropic layer, d/d(angle) at 0
        angle_uniz  uniaxial-z layer (exx=eyy != ezz), d/d(angle) at 0
        phi_anis    in-plane anisotropic layer, d/d(phi) at angle 0.3, phi 0
        For in-plane tensors Delta depends on (Kx, Ky) only through Kx^2,
        Ky^2, Kx Ky, so d/d(angle) at 0 is exactly 0; an isotropic medium's
        s/p pairs stay degenerate at EVERY (angle, phi) -- never split.
Outputs: R(2), T(2), Re/Im Jr (8) = 12.
"""
from _dcommon import capture, gauge, parity, sweep
from _h import dump, jax, jnp, np

from lumenairy.elements.berreman import berreman_jones_1d

WL, T1, T2 = 1e-6, 0.3e-6, 0.1e-6
E0 = 2.56
XS = {"exy": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], complex),
      "xxyy": np.diag([1.0, -1.0, 0.0]).astype(complex),
      "iso": np.eye(3, dtype=complex)}


def pack(R, T, Jr, xp):
    Jr = xp.ravel(Jr)
    return xp.concatenate([xp.ravel(R), xp.ravel(T), xp.real(Jr),
                           xp.imag(Jr)])


def f_delta(X, xp):
    def f(d):
        e = E0 * xp.eye(3, dtype=complex) + d * xp.asarray(X)
        if xp is jnp:
            R, T, Jr, _Jt = berreman_jones_1d(
                [(e, jnp.asarray(T1)), (jnp.asarray(2.1 + 0j),
                                        jnp.asarray(T2))],
                jnp.asarray(1.0 + 0j), jnp.asarray(1.45 + 0j),
                jnp.asarray(WL))
        else:
            R, T, Jr, _Jt = berreman_jones_1d([(e, T1), (2.1, T2)], 1.0,
                                              1.45, WL)
        return pack(R, T, Jr, xp)
    return f


# CONCRETE (closed-over) tensors so the traced call stays on ``_solve_jax``
# (a (3,3) tensor built inside jit is a tracer and routes to the off-plane
# solve, see above).
EPS_ISO = E0 * np.eye(3, dtype=complex)
EPS_UNIZ = np.diag([2.56, 2.56, 3.0]).astype(complex)     # uniaxial, axis z
_c, _s = np.cos(0.4), np.sin(0.4)
EPS_ANIS = np.array([[2.56 * _c * _c + 2.25 * _s * _s, 0.31 * _c * _s, 0],
                     [0.31 * _c * _s, 2.56 * _s * _s + 2.25 * _c * _c, 0],
                     [0, 0, 2.25]], complex)               # in-plane, exy != 0
J_CONST = {k: jnp.asarray(v) for k, v in (("iso", EPS_ISO),
                                          ("uniz", EPS_UNIZ),
                                          ("anis", EPS_ANIS))}
NP_CONST = {"iso": EPS_ISO, "uniz": EPS_UNIZ, "anis": EPS_ANIS}


def f_angle(xp, which="iso", var="angle", fixed=0.0):
    """d/d(angle) (phi = fixed) or d/d(phi) (angle = fixed)."""
    def f(a):
        ang, phi = (a, fixed) if var == "angle" else (fixed, a)
        if xp is jnp:
            R, T, Jr, _Jt = berreman_jones_1d(
                [(J_CONST[which], jnp.asarray(T1)),
                 (jnp.asarray(2.1 + 0j), jnp.asarray(T2))],
                jnp.asarray(1.0 + 0j), jnp.asarray(1.45 + 0j),
                jnp.asarray(WL), angle=jnp.asarray(ang), phi=jnp.asarray(phi))
        else:
            R, T, Jr, _Jt = berreman_jones_1d(
                [(NP_CONST[which], T1), (2.1, T2)], 1.0, 1.45, WL,
                angle=float(ang), phi=float(phi))
        return pack(R, T, Jr, xp)
    return f


ANG_CASES = {"angle_iso": ("iso", "angle", 0.0, (0.0, 1e-3)),
             "angle_uniz": ("uniz", "angle", 0.0, (0.0, 1e-3)),
             "phi_anis_at_angle0.3": ("anis", "phi", 0.3, (0.0, 0.2))}


out = {"spectrum": {}, "parity": {}, "sweep": {}, "gauge": {}}
for name, X in XS.items():
    for d in (0.0, 1e-5, 1e-3):
        sp = capture(lambda X=X, d=d: jax.jit(f_delta(X, jnp))(
            jnp.asarray(d)))
        out["spectrum"][f"{name}_{d!r}"] = sp
        print("spectrum", name, d, sp, flush=True)
for cname, (w, var, fixed, xs) in ANG_CASES.items():
    for a in xs:
        sp = capture(lambda a=a, w=w, var=var, fixed=fixed: jax.jit(
            f_angle(jnp, w, var, fixed))(jnp.asarray(a)))
        out["spectrum"][f"{cname}_{a!r}"] = sp
        print("spectrum", cname, a, sp, flush=True)

for name, X in XS.items():
    fj, fn = f_delta(X, jnp), f_delta(X, np)
    out["parity"][name] = parity(fj, fn, 0.0)
    out["sweep"][name] = sweep(fj, fn, (0.0, 1e-5, 1e-3), label=name)
    out["gauge"][name] = gauge(fj, 0.0)
    print("parity", out["parity"][name], "gauge", out["gauge"][name],
          flush=True)

for cname, (w, var, fixed, xs) in ANG_CASES.items():
    fj, fn = f_angle(jnp, w, var, fixed), f_angle(np, w, var, fixed)
    out["parity"][cname] = parity(fj, fn, xs[0])
    out["sweep"][cname] = sweep(fj, fn, xs, label=cname)
    out["gauge"][cname] = gauge(fj, xs[0]) if np.any(np.asarray(
        out["sweep"][cname][repr(xs[0])]["AD"]) != 0) else "AD == 0"
    print("parity", cname, out["parity"][cname], "gauge",
          out["gauge"][cname], flush=True)
print(dump("d4_berreman.json", out))

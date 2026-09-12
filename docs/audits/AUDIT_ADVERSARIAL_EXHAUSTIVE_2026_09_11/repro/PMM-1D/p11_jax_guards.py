"""PROBE 11: the DIFFERENTIABLE (JAX) PMMStack twin has NONE of the NumPy
path's guards.  Measure what it returns where NumPy refuses.

NumPy `PMMStack.solve` runs, in order:
  _require_propagating_incidence, _guarded_lstsq (M1 Rayleigh projection),
  _guarded_inverse (interface + Redheffer stars), _warn_stack_energy
  (non-finite raise / negative raise / R+T>1 warn), the sliver guard.
`_jax_stack.py` contains none of those strings.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from lumenairy.elements.pmm import PMMStack

per, wl = 1.0e-6, 1.55e-6
eps_hi, eps_lo = 3.48 ** 2, 1.444 ** 2


def mk(traced, *, degree, ffo, segs2=None, n_sup=1.0, angle=0.2,
       thick=(0.25e-6, 0.25e-6)):
    st = PMMStack(per, n_substrate=1.444, n_superstrate=n_sup, degree=degree,
                  far_field_orders=ffo)
    eh = jnp.asarray(eps_hi + 0j) if traced else eps_hi
    st.add_layer(thick[0], segments=[(0.5, eh), (0.5, eps_lo)])
    if segs2 is not None:
        st.add_layer(thick[1], segments=segs2)
    st.set_source(wl, angle=angle)
    return st


def run(st):
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        try:
            o, R, T, J = st.solve()
            return ("ok", np.asarray(R), np.asarray(T), np.asarray(J),
                    [str(w.message)[:70] for w in W])
        except Exception as e:
            return ("RAISE " + type(e).__name__, None, None, None,
                    [str(e)[:150]])


print("=== 11a: OVER-CAPACITY Rayleigh projection (the M1 _guarded_lstsq "
      "case) ===")
print(f"{'degree':>7} {'ffo':>5} | {'numpy':>50} | {'jax':>40}")
for degree, ffo in ((4, 41), (5, 41), (6, 41), (8, 61), (4, 9), (6, 21)):
    sn = run(mk(False, degree=degree, ffo=ffo))
    sj = run(mk(True, degree=degree, ffo=ffo))
    def brief(s):
        if s[0] != "ok":
            return s[0]
        tot = s[1].sum(1) + s[2].sum(1)
        return f"tot={np.array2string(tot, precision=6)}"
    print(f"{degree:7d} {ffo:5d} | {brief(sn):>50} | {brief(sj):>40}")
    if sn[0] == "ok" and sj[0] == "ok":
        print(f"{'':15}   max|dR| numpy-vs-jax = "
              f"{np.max(np.abs(sn[1]-sj[1])):.3e}")

print()
print("=== 11b: a stack whose NumPy solve is REFUSED by the sliver guard ===")
s = 1.5e-5
segs2 = [(0.5 + s, eps_hi), (0.5 - s, eps_lo)]
for degree in (12, 14, 16, 18, 20):
    sn = run(mk(False, degree=degree, ffo=11, segs2=segs2))
    sj = run(mk(True, degree=degree, ffo=11, segs2=segs2))
    def tag(x):
        if x[0] != "ok":
            return x[0]
        tot = x[1].sum(1) + x[2].sum(1)
        return (f"T0(Ey)={x[2][1, len(x[2][1])//2]:.7f} "
                f"tot={float(tot.max()):.6f}")
    print(f"  deg={degree:3d}  numpy: {tag(sn):45s}  jax: {tag(sj)}")

print()
print("=== 11c: NaN / gain / grazing on the JAX path ===")
for lbl, kw in (("NaN eps", dict(degree=12, ffo=11)),
                ("gain n_sup", dict(degree=12, ffo=11, n_sup=1.0 - 1e-3j)),
                ("near-grazing", dict(degree=12, ffo=11, angle=1.5707))):
    if lbl == "NaN eps":
        stn = mk(False, **kw)
        stn._layers[0] = (stn._layers[0][0],
                          [(0.5, np.nan * np.eye(3)), (0.5, eps_lo * np.eye(3))],
                          0.0)
        stj = mk(True, **kw)
        stj._layers[0] = (stj._layers[0][0],
                          [(0.5, jnp.asarray(np.nan * np.eye(3) + 0j)),
                           (0.5, jnp.asarray(eps_lo * np.eye(3) + 0j))], 0.0)
    else:
        stn, stj = mk(False, **kw), mk(True, **kw)
    rn, rj = run(stn), run(stj)
    def t(x):
        if x[0] != "ok":
            return x[0] + " :: " + x[4][0][:80]
        return f"tot={np.array2string(x[1].sum(1)+x[2].sum(1), precision=5)}"
    print(f"  {lbl:14s} numpy: {t(rn)}")
    print(f"  {'':14s} jax  : {t(rj)}")

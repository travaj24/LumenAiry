"""PROBE 4: S-matrix stack -- thick lossy layer stability, TMM parity,
and the JAX twin (x64 + gradient finiteness).
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/PMM-1D")
from oracle import tmm_stack
from lumenairy.elements.pmm import PMMStack

wl = 1.0e-6
period = 0.3e-6                      # subwavelength: 0 order only

print("=== 4a: THICK LOSSY uniform layers (no T-matrix growth?) ===")
for thick in (1e-6, 10e-6, 50e-6, 200e-6):
    for kappa in (0.0, 0.01, 0.3):
        st = PMMStack(period, n_substrate=1.5, n_superstrate=1.0, degree=8,
                      far_field_orders=5)
        nl = 2.0 + 1j * kappa
        st.add_layer(thick, eps=nl ** 2)
        st.set_source(wl, angle=np.deg2rad(20.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
        m0 = int(np.where(o == 0)[0][0])
        # oracle
        rs, ts, Rs, Ts = tmm_stack([1.0, nl, 1.5], [thick], wl,
                                   np.deg2rad(20.0), 's')
        rp, tp, Rp, Tp_ = tmm_stack([1.0, nl, 1.5], [thick], wl,
                                    np.deg2rad(20.0), 'p')
        # row 1 = incident Ey = s ; row 0 = incident Ex = p
        print(f"L={thick*1e6:7.1f}um k={kappa:4.2f}: "
              f"s R {R[1,m0]:.10f}/{Rs:.10f} T {T[1,m0]:.10f}/{Ts:.10f} | "
              f"p R {R[0,m0]:.10f}/{Rp:.10f} T {T[0,m0]:.10f}/{Tp_:.10f}")

print()
print("=== 4b: DEEP GRATING stack -- energy + convergence ===")
per = 1.0e-6
for nlay in (1, 4, 12, 40):
    st = PMMStack(per, n_substrate=1.444, n_superstrate=1.0, degree=20,
                  far_field_orders=15)
    for _ in range(nlay):
        st.add_layer(0.45e-6 / nlay,
                     segments=[(0.5, 3.48 ** 2), (0.5, 1.444 ** 2)])
    st.set_source(1.55e-6, angle=np.deg2rad(17.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve()
    tot = R.sum(axis=1) + T.sum(axis=1)
    print(f"nlay={nlay:3d}: tot = {tot}  J00 = {J[0,0]:.10f} {J[1,1]:.10f}")

print()
print("=== 4c: 50um ABSORBING grating layer (S-matrix stability) ===")
st = PMMStack(1.0e-6, n_substrate=1.444, n_superstrate=1.0, degree=16,
              far_field_orders=11)
st.add_layer(50e-6, segments=[(0.5, (3.48 + 0.05j) ** 2), (0.5, 1.444 ** 2)])
st.set_source(1.55e-6, angle=np.deg2rad(10.0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    o, R, T, J = st.solve()
print("tot =", R.sum(axis=1) + T.sum(axis=1), " finite:",
      np.all(np.isfinite(R)) and np.all(np.isfinite(T)))
print("max|R|", R.max(), "max|T|", T.max())

print()
print("=== 4d: JAX twin parity + grad finiteness ===")
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def build(eps_r):
        st = PMMStack(1.0e-6, n_substrate=1.444, n_superstrate=1.0, degree=14,
                      far_field_orders=11)
        st.add_layer(0.45e-6, segments=[(0.5, eps_r), (0.5, 1.444 ** 2)])
        st.set_source(1.55e-6, angle=0.2)
        return st.solve()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_n, R_n, T_n, J_n = build(3.48 ** 2 + 0j)
        o_j, R_j, T_j, J_j = build(jnp.asarray(3.48 ** 2 + 0j))
    print("parity maxdR =", float(np.max(np.abs(np.asarray(R_j) - R_n))),
          " maxdT =", float(np.max(np.abs(np.asarray(T_j) - T_n))),
          " maxdJ =", float(np.max(np.abs(np.asarray(J_j) - J_n))))

    def f(e):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = build(e)
        return jnp.real(T[1]).sum()

    g = jax.grad(f, holomorphic=False)(jnp.asarray(3.48 ** 2 + 0.0j))
    print("grad wrt eps_ridge =", g, " finite:", bool(np.isfinite(np.abs(g))))
    # finite difference
    h = 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fp = f(jnp.asarray(3.48 ** 2 + h + 0j))
        fm = f(jnp.asarray(3.48 ** 2 - h + 0j))
    print("FD =", float((fp - fm) / (2 * h)), " AD(real part) =",
          float(np.real(g)))
except Exception as e:
    print("JAX probe failed:", type(e).__name__, e)

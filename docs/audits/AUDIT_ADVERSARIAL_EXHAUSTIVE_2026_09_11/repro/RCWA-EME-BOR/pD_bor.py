import numpy as np, warnings, time
t0 = time.perf_counter()
from lumenairy.elements.bor import BORStack
from lumenairy.elements.bor.fiber_oracle import fiber_modes, fiber_det
from lumenairy.elements.bor.stepindex_oracle import stepindex_modes
import lumenairy.elements.bor.coupled_radial_eigensolver as cre
import lumenairy.elements.bor as borpkg
print("import %.1f s" % (time.perf_counter() - t0), flush=True)
print("bor exports:", sorted(n for n in dir(borpkg) if not n.startswith('_')),
      flush=True)

# ---------------------------------------------------------------- oracle audit
print("\n===== O1) AUDIT THE ORACLE: exact hybrid vs LP (weakly guiding) =====",
      flush=True)
# V = 2.4, n_core=1.45, n_clad=1.44  ->  a from V = k0 a sqrt(n1^2-n2^2)
lam = 1.55e-6
k0 = 2*np.pi/lam
n1, n2 = 1.45, 1.44
V = 2.4
a = V/(k0*np.sqrt(n1**2 - n2**2))
print(f"   lam={lam:.3e} a={a:.6e} m  NA={np.sqrt(n1**2-n2**2):.6f}", flush=True)


def lp_char(b, l, V):
    """LP_{l,m} characteristic equation residual (weakly guiding):
    U J_{l-1}(U)/J_l(U) = -W K_{l-1}(W)/K_l(W), b=(neff^2-n2^2)/(n1^2-n2^2)."""
    from scipy.special import jv, kv
    U = V*np.sqrt(1-b)
    W = V*np.sqrt(b)
    return U*jv(l-1, U)/jv(l, U) + W*kv(l-1, W)/kv(l, W)


from scipy.optimize import brentq
# LP01 (l=0 uses J_1/J_0 form): U J1(U)/J0(U) = W K1(W)/K0(W)


def lp01(b, V):
    from scipy.special import jv, kv
    U = V*np.sqrt(1-b); W = V*np.sqrt(b)
    return U*jv(1, U)/jv(0, U) - W*kv(1, W)/kv(0, W)


blo, bhi = 1e-9, 1-1e-9
bs = np.linspace(blo, bhi, 20001)
vals = np.array([lp01(b, V) for b in bs])
roots = []
for i in range(len(bs)-1):
    if np.isfinite(vals[i]) and np.isfinite(vals[i+1]) and vals[i]*vals[i+1] < 0:
        try:
            roots.append(brentq(lp01, bs[i], bs[i+1], args=(V,), xtol=1e-15,
                                rtol=8.9e-16))
        except Exception:
            pass
b_lp = max(roots) if roots else np.nan
neff_lp = np.sqrt(n2**2 + b_lp*(n1**2-n2**2))
print(f"   LP01 (weakly-guiding):  b={b_lp:.12f}  neff={neff_lp:.12f}",
      flush=True)
# exact HE11 from the library's fiber oracle (m=1)
qs = fiber_modes(1, a, n1**2, n2**2, k0)
print(f"   exact m=1 oracle q/k0 = {np.array(qs)/k0}", flush=True)
if len(qs):
    print(f"   |neff_exact(HE11) - neff_LP01| = "
          f"{abs(qs[0]/k0 - neff_lp):.3e}  (relative "
          f"{abs(qs[0]/k0 - neff_lp)/neff_lp:.3e})", flush=True)
for m in (0, 2):
    qq = fiber_modes(m, a, n1**2, n2**2, k0)
    print(f"   exact m={m} oracle q/k0 = {np.array(qq)/k0}", flush=True)

print("\n===== O2) coupled radial eigensolver vs the FIBER oracle =====",
      flush=True)
fns = [n for n in dir(cre) if not n.startswith('_')]
print("   cre exports:", fns, flush=True)
try:
    import inspect
    for nm in ("coupled_modes", "solve_modes", "modes"):
        if hasattr(cre, nm):
            print("   sig", nm, inspect.signature(getattr(cre, nm)), flush=True)
except Exception as e:
    print("   sig err", e, flush=True)

print("\n===== O3) BOR reproducer (AUDIT_BOR_PROPAGATING_CUTOFF) =====",
      flush=True)
try:
    S = 1e6
    LAM = 1.0e-6
    k0r = 2.0*np.pi/(LAM*S)
    s = BORStack(Rbig=48e-6*S, m=1, N=256, n_superstrate=1.41+0j,
                 n_substrate=1.41+0j)
    s.add_layer(0.5e-6*S, rings=(3.0e-6*S, 0.5, 2.45+0j, 1.41+0j))
    s.set_source(k0=k0r)
    t = time.perf_counter()
    res = s.solve()
    e = np.asarray(res["energy"], float)
    print(f"   ({time.perf_counter()-t:.1f}s) n_modes={e.size} "
          f"max|E-1|={float(np.max(np.abs(e-1.0))):.6e}", flush=True)
    print(f"   R[0]={float(np.asarray(res['R'])[0]):.6f} "
          f"T[0]={float(np.asarray(res['T'])[0]):.6f}", flush=True)
    keys = list(res.keys())
    print("   keys:", keys, flush=True)
    if "q" in res:
        qn = np.asarray(res["q"])/k0r
        print(f"   min q/k0 (kept) = {np.min(np.real(qn)):.6f}", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n===== O4) unit-scale invariance of the reproducer =====", flush=True)
for S in (1.0, 1e6, 1e9):
    try:
        LAM = 1.0e-6
        k0r = 2.0*np.pi/(LAM*S)
        s = BORStack(Rbig=48e-6*S, m=1, N=128, n_superstrate=1.41+0j,
                     n_substrate=1.41+0j)
        s.add_layer(0.5e-6*S, rings=(3.0e-6*S, 0.5, 2.45+0j, 1.41+0j))
        s.set_source(k0=k0r)
        res = s.solve()
        e = np.asarray(res["energy"], float)
        print(f"   S={S:g}: n={e.size} max|E-1|={float(np.max(np.abs(e-1))):.6e}"
              f" R0={float(np.asarray(res['R'])[0]):.10f}", flush=True)
    except Exception as ex:
        print(f"   S={S:g} RAISED {type(ex).__name__}: {str(ex)[:160]}",
              flush=True)

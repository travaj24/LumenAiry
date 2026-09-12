import numpy as np, warnings, time
t0 = time.perf_counter()
from lumenairy.elements.bor import (BORStack, radial_coupled_modes,
                                    guided_modes, fiber_modes, layer_modes,
                                    radial_spectrum)
import inspect
print("import %.1f s" % (time.perf_counter() - t0), flush=True)
print("  radial_coupled_modes", inspect.signature(radial_coupled_modes),
      flush=True)
print("  guided_modes", inspect.signature(guided_modes), flush=True)
print("  radial_spectrum", inspect.signature(radial_spectrum), flush=True)

print("\n=== B1) coupled radial eigensolver vs the fiber oracle ===", flush=True)
lam = 1.55
k0 = 2*np.pi/lam
n1, n2 = 1.45, 1.44
V = 2.4
a = V/(k0*np.sqrt(n1**2-n2**2))
Rb = 8*a
print(f"   a={a:.6f}  Rbig={Rb:.6f}  (V={V})", flush=True)
q_or = fiber_modes(1, a, n1**2, n2**2, k0)
print(f"   oracle HE11 q/k0 = {np.array(q_or)/k0}", flush=True)


def eps_prof(r):
    return np.where(r < a, n1**2, n2**2).astype(complex)


for N in (200, 400, 800):
    for stag in (True, False):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = guided_modes(1, Rb, N, eps_prof, k0)
            q = np.asarray(out[0] if isinstance(out, tuple) else out)
            q = np.atleast_1d(q)
            qr = np.real(q)
            band = qr[(qr > n2*k0) & (qr < n1*k0)]
            best = np.max(band)/k0 if band.size else float('nan')
            print(f"   N={N:4d} guided_modes: n_in_band={band.size} "
                  f"best neff={best:.10f} err={best-q_or[0]/k0:+.3e}", flush=True)
            break
        except Exception as e:
            print(f"   N={N} guided_modes RAISED {type(e).__name__}: "
                  f"{str(e)[:200]}", flush=True)
            break

print("\n=== B2) BOR reproducer: FUNDAMENTAL (argmax q) R pin ===", flush=True)
S = 1e6
LAM = 1.0e-6
k0r = 2.0*np.pi/(LAM*S)
s = BORStack(Rbig=48e-6*S, m=1, N=256, n_superstrate=1.41+0j,
             n_substrate=1.41+0j)
s.add_layer(0.5e-6*S, rings=(3.0e-6*S, 0.5, 2.45+0j, 1.41+0j))
s.set_source(k0=k0r)
res = s.solve()
e = np.asarray(res["energy"], float)
q = np.asarray(res["q"])
R = np.asarray(res["R"])
j = int(np.argmax(np.real(q)))
print(f"   n_modes={e.size} max|E-1|={float(np.max(np.abs(e-1))):.6e}",
      flush=True)
print(f"   fundamental (argmax q) index={j} q/k0={np.real(q[j])/k0r:.6f} "
      f"R={R[j]:.6f}  [doc claims 0.142290 post-anchor-flip]", flush=True)
print(f"   min q/k0 kept = {np.min(np.real(q))/k0r:.6f}  "
      f"[doc claims 0.0512]", flush=True)

print("\n=== B3) unit-scale invariance at S=1e9 ===", flush=True)
for Sx in (1.0, 1e6, 1e9):
    try:
        k0x = 2.0*np.pi/(1.0e-6*Sx)
        st = BORStack(Rbig=48e-6*Sx, m=1, N=96, n_superstrate=1.41+0j,
                      n_substrate=1.41+0j)
        st.add_layer(0.5e-6*Sx, rings=(3.0e-6*Sx, 0.5, 2.45+0j, 1.41+0j))
        st.set_source(k0=k0x)
        rr = st.solve()
        ee = np.asarray(rr["energy"], float)
        qq = np.asarray(rr["q"])
        jj = int(np.argmax(np.real(qq)))
        print(f"   S={Sx:g}: n={ee.size} max|E-1|={np.max(np.abs(ee-1)):.3e} "
              f"R_fund={np.asarray(rr['R'])[jj]:.12f}", flush=True)
    except Exception as ex:
        print(f"   S={Sx:g} RAISED {type(ex).__name__}: {str(ex)[:200]}",
              flush=True)

print("\n=== B4) BOR reciprocity/symmetry: S-matrix reciprocity ===", flush=True)
try:
    Smat = res["S"]
    print("   S type:", type(Smat), flush=True)
    if isinstance(Smat, (tuple, list)):
        S11, S12, S21, S22 = Smat
        print(f"   |S21 - S12^T| / |S21| = "
              f"{np.max(np.abs(np.asarray(S21)-np.asarray(S12).T))/max(np.max(np.abs(S21)),1e-30):.3e}",
              flush=True)
        print(f"   |S11 - S11^T| / |S11| = "
              f"{np.max(np.abs(np.asarray(S11)-np.asarray(S11).T))/max(np.max(np.abs(S11)),1e-30):.3e}",
              flush=True)
except Exception as ex:
    print("   ", type(ex).__name__, str(ex)[:200], flush=True)

print("\n=== B5) cache poisoning re-check ===", flush=True)
from lumenairy.elements.rcwa import RCWAStack
from lumenairy.elements.rcwa import _core as RC
RC._clear_rcwa_caches()
st = RCWAStack(0.5e-6, period_y=0.5e-6, n_superstrate=1.0, n_substrate=1.5,
               n_orders=3, n_orders_y=3)
st.add_layer(0.2e-6, eps=4.0)
st.set_source(0.633e-6, theta=0.2)
r1 = st.solve()
a1 = r1.per_order_amplitudes()
kz_before = np.array(a1["kz"], copy=True)
a1["kz"] *= 2.0
r2 = st.solve()
a2 = r2.per_order_amplitudes()
print(f"   max|kz(after) - kz(before)| = "
      f"{np.max(np.abs(a2['kz']-kz_before)):.3e}  (0 => cache NOT poisoned)",
      flush=True)
# Ex aliasing within the same result object
a1["Ex"][:] = 0.0
a3 = r1.per_order_amplitudes()
print(f"   same-result Ex aliasing: max|Ex| after zeroing = "
      f"{np.max(np.abs(a3['Ex'])):.3e}  (0 => aliased/mutable)", flush=True)

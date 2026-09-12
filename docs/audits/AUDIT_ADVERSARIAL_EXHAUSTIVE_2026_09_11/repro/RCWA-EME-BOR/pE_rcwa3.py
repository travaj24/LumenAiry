import numpy as np, warnings, time
t0 = time.perf_counter()
from lumenairy.elements.rcwa import (rcwa_efficiency_1d, rcwa_efficiency_2d,
                                     rcwa_jones_2d)
print("import %.1f s" % (time.perf_counter() - t0), flush=True)

Sx = 96
xg = np.arange(Sx)/Sx
X, Y = np.meshgrid(xg, xg, indexing='ij')
cell_sq = np.where((np.abs(X-0.5) < 0.25) & (np.abs(Y-0.5) < 0.25), 6.25, 2.25)
cell_disk = np.where(((X-0.5)**2 + (Y-0.5)**2) < 0.2**2, 6.25, 2.25)
print("   cell transpose-symmetric:",
      np.array_equal(cell_sq, cell_sq.T), np.array_equal(cell_disk, cell_disk.T),
      flush=True)


def tensorize(cell):
    T = np.zeros(cell.shape + (3, 3), dtype=complex)
    for i in range(3):
        T[..., i, i] = cell
    return T


print("\n=== K) C4 symmetry violation of rcwa_jones_2d vs truncation ===",
      flush=True)
for name, cell in (("square", cell_sq), ("disk", cell_disk)):
    for form in ("laurent", "li", "fff_nv"):
        row = []
        for M in (4, 6, 8, 10, 12):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = rcwa_jones_2d(0.5e-6, 0.5e-6, tensorize(cell), 1.5,
                                        1.0, 0.3e-6, 0.633e-6, theta=0.0,
                                        phi=0.0, n_orders_x=M, n_orders_y=M,
                                        formulation=form)
                J = out[3]
                row.append((M, abs(J[0, 0]-J[1, 1]), J[0, 0]))
            except Exception as e:
                row.append((M, float('nan'), str(e)[:60]))
        print(f"   {name:7s} {form:8s}: " +
              "  ".join(f"M={m}:|dJ|={d:.2e}" for m, d, _ in row), flush=True)
        print(f"            Jxx: " +
              "  ".join(f"{(j if isinstance(j, complex) else 0):.8f}"
                        for _, _, j in row), flush=True)

print("\n=== L) 2-D separable limit vs 1-D, GRID-EXACT duty ===", flush=True)
duty = 0.5
prof = np.where(xg < duty, 4.0, 1.0)
cell_1d = np.repeat(prof[:, None], Sx, axis=1)
for pol in ("te", "tm"):
    for ang in (0.0, 30.0):
        o1, R1, T1 = rcwa_efficiency_1d(0.5e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6,
                                        duty, 0.633e-6, angle=np.deg2rad(ang),
                                        polarization=pol, n_orders=8,
                                        formulation='li')
        for form in ("laurent", "li"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell_1d, 1.5, 1.0,
                                       0.3e-6, 0.633e-6,
                                       theta=np.deg2rad(ang), phi=0.0,
                                       polarization=pol, n_orders_x=8,
                                       n_orders_y=0, formulation=form)
            oo, R2, T2 = np.asarray(r[0]), np.asarray(r[1]), np.asarray(r[2])
            sel = {int(v[0]): i for i, v in enumerate(oo)}
            dR = max(abs(R1[8+m] - R2[sel[m]]) for m in (-1, 0, 1) if m in sel)
            dT = max(abs(T1[8+m] - T2[sel[m]]) for m in (-1, 0, 1) if m in sel)
            print(f"   pol={pol} ang={ang:4.1f} {form:8s}: max|dR|={dR:.3e} "
                  f"max|dT|={dT:.3e}", flush=True)

print("\n=== M) Li vs Laurent metallic-TM convergence (1-D) ===", flush=True)
args = dict(period=0.5e-6, n_ridge=0.135+3.99j, n_groove=1.0, n_substrate=1.5,
            n_superstrate=1.0, depth=0.2e-6, duty_cycle=0.5,
            wavelength=0.6328e-6)
print("%6s %18s %18s %13s %13s" % ("N", "R0(li)", "R0(laurent)", "A(li)",
                                   "A(laurent)"), flush=True)
for M in (5, 11, 21, 41, 61, 81, 101, 151, 201):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, Rl, Tl = rcwa_efficiency_1d(**args, polarization='tm', n_orders=M,
                                       formulation='li')
        o, Ra, Ta = rcwa_efficiency_1d(**args, polarization='tm', n_orders=M,
                                       formulation='laurent')
    print("%6d %18.12f %18.12f %13.5e %13.5e"
          % (2*M+1, Rl[M], Ra[M], 1-Rl.sum()-Tl.sum(), 1-Ra.sum()-Ta.sum()),
          flush=True)

print("\n=== N) ASR (Granet matched coordinates) accuracy claim ===", flush=True)
ref = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    o, Rr, Tr = rcwa_efficiency_1d(**args, polarization='tm', n_orders=301,
                                   formulation='li')
ref = Rr[301]
print(f"   reference R0 at N=603 (li, uniform) = {ref:.12f}", flush=True)
for M in (6, 12, 24, 48):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, Ru, Tu = rcwa_efficiency_1d(**args, polarization='tm', n_orders=M,
                                       formulation='li')
        best = None
        for eta in (0.3, 0.5, 0.7, 0.8):
            try:
                o, Ra2, Ta2 = rcwa_efficiency_1d(**args, polarization='tm',
                                                 n_orders=M, formulation='li',
                                                 asr_eta=eta)
                e = abs(Ra2[M]-ref)
                if best is None or e < best[1]:
                    best = (eta, e, Ra2[M])
            except Exception as ex:
                pass
    print(f"   M={M:3d}: uniform err={abs(Ru[M]-ref):.3e}  "
          f"ASR best eta={best[0] if best else None} err="
          f"{best[1] if best else float('nan'):.3e}", flush=True)

print("\n=== P) rcwa_convergence / rcwa_extrapolate sanity ===", flush=True)
from lumenairy.elements.rcwa import rcwa_extrapolate
vals = [Rl[M] for M in ()] if False else None
try:
    seq = []
    for M in (20, 40, 80):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, Rq, Tq = rcwa_efficiency_1d(**args, polarization='tm',
                                           n_orders=M, formulation='li')
        seq.append(Rq[M])
    ex = rcwa_extrapolate(seq, n_orders=[20, 40, 80])
    print(f"   seq={seq}  extrapolated={ex}  ref={ref:.12f}  "
          f"err_extrap={abs(np.ravel(np.asarray(ex,dtype=object))[0]-ref) if not isinstance(ex,tuple) else 'tuple'}",
          flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()

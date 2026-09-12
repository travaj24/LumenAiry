import sys, numpy as np, warnings, time
t0 = time.perf_counter()
from lumenairy.elements.rcwa import (rcwa_efficiency_1d, rcwa_efficiency_2d,
                                     rcwa_jones_2d)
print("import %.1f s" % (time.perf_counter() - t0), flush=True)


def fmt(z):
    return f"{abs(z):.10f}<{np.angle(z):+.9f}"


print("\n===== D3) reciprocity residual vs truncation =====", flush=True)
args = dict(period=1.3e-6, n_ridge=2.1, n_groove=1.0, depth=0.45e-6,
            duty_cycle=0.4, wavelength=0.633e-6)
nI, nII = 1.0, 1.62
th = np.deg2rad(18.0)
wl0 = args['wavelength']; Pp = args['period']
for M in (11, 21, 41, 81):
    for pol in ('te', 'tm'):
        o, R, T = rcwa_efficiency_1d(n_substrate=nII, n_superstrate=nI,
                                     angle=th, polarization=pol, n_orders=M,
                                     **args)
        worst = 0.0
        for m in (-2, -1, 1, 2):
            s_m = (nI*np.sin(th) + m*wl0/Pp)/nII
            if abs(s_m) >= 1:
                continue
            th_m = np.arcsin(s_m)
            o2, R2, T2 = rcwa_efficiency_1d(n_substrate=nI, n_superstrate=nII,
                                            angle=-th_m, polarization=pol,
                                            n_orders=M, **args)
            worst = max(worst, abs(T[M+m]-T2[M+m])/max(T[M+m], 1e-30))
        print(f"   M={M:3d} pol={pol}: worst rel reciprocity residual = "
              f"{worst:.3e}", flush=True)

Sx = 64
xg = np.arange(Sx)/Sx
X, Y = np.meshgrid(xg, xg, indexing='ij')
cell_sq = np.where((np.abs(X-0.5) < 0.15) & (np.abs(Y-0.5) < 0.15), 6.25, 2.25)
cell_disk = np.where(((X-0.5)**2 + (Y-0.5)**2) < 0.15**2, 6.25, 2.25)


def tensorize(cell):
    T = np.zeros(cell.shape + (3, 3), dtype=complex)
    for i in range(3):
        T[..., i, i] = cell
    return T


print("\n===== F) 2-D symmetry tests (normal incidence) =====", flush=True)
for name, cell in (("square-C4", cell_sq), ("disk", cell_disk)):
    for form in ("laurent", "li", "fff_nv"):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore")
                out = rcwa_jones_2d(0.5e-6, 0.5e-6, tensorize(cell), 1.5, 1.0,
                                    0.3e-6, 0.633e-6, theta=0.0, phi=0.0,
                                    n_orders_x=6, n_orders_y=6,
                                    formulation=form)
            J = out[3]
            print(f"   {name:10s} {form:8s}: Jxx={fmt(J[0,0])} Jyy={fmt(J[1,1])}"
                  f" |Jxx-Jyy|={abs(J[0,0]-J[1,1]):.3e} |Jxy|={abs(J[0,1]):.3e}"
                  f" |Jyx|={abs(J[1,0]):.3e}", flush=True)
        except Exception as e:
            print(f"   {name} {form} RAISED {type(e).__name__}: {str(e)[:200]}",
                  flush=True)

print("\n===== G) phi-rotation covariance of a C-inf (disk) cell =====",
      flush=True)
th2 = np.deg2rad(20.0)
J0 = None
for phid in (0.0, 30.0, 90.0):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rcwa_jones_2d(0.5e-6, 0.5e-6, tensorize(cell_disk), 1.5, 1.0,
                                0.3e-6, 0.633e-6, theta=th2,
                                phi=np.deg2rad(phid), n_orders_x=6,
                                n_orders_y=6, formulation="li")
        J = out[3]
        print(f"   phi={phid:5.1f}: J=[[{fmt(J[0,0])},{fmt(J[0,1])}],"
              f"[{fmt(J[1,0])},{fmt(J[1,1])}]]", flush=True)
        if phid == 0.0:
            J0 = J.copy()
        else:
            c, s = np.cos(np.deg2rad(phid)), np.sin(np.deg2rad(phid))
            Rm = np.array([[c, -s], [s, c]])
            print(f"      R J0 R^T residual = "
                  f"{np.max(np.abs(J - Rm @ J0 @ Rm.T)):.3e}", flush=True)
    except Exception as e:
        print(f"   phi={phid} RAISED {type(e).__name__}: {str(e)[:200]}",
              flush=True)

print("\n===== H) 2-D separable (y-uniform) limit vs 1-D solver =====",
      flush=True)
duty = 0.4
prof = np.where(xg < duty, 4.0, 1.0)
cell_1d = np.repeat(prof[:, None], Sx, axis=1)
for pol in ("te", "tm"):
    for ang in (0.0, 30.0):
        o1, R1, T1 = rcwa_efficiency_1d(0.5e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6,
                                        duty, 0.633e-6, angle=np.deg2rad(ang),
                                        polarization=pol, n_orders=8,
                                        formulation='li')
        for form in ("laurent", "li"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell_1d, 1.5, 1.0,
                                           0.3e-6, 0.633e-6,
                                           theta=np.deg2rad(ang), phi=0.0,
                                           polarization=pol, n_orders_x=8,
                                           n_orders_y=0, formulation=form)
                oo, R2, T2 = r[0], np.asarray(r[1]), np.asarray(r[2])
                oo = np.asarray(oo)
                sel = {int(v[0]): i for i, v in enumerate(oo)}
                dR = max(abs(R1[8+m] - R2[sel[m]]) for m in (-1, 0, 1)
                         if m in sel)
                dT = max(abs(T1[8+m] - T2[sel[m]]) for m in (-1, 0, 1)
                         if m in sel)
                print(f"   pol={pol} ang={ang:4.1f} {form:8s}: max|dR|={dR:.3e}"
                      f" max|dT|={dT:.3e} closure={R2.sum()+T2.sum()-1:+.3e}",
                      flush=True)
            except Exception as e:
                print(f"   pol={pol} ang={ang} {form} RAISED "
                      f"{type(e).__name__}: {str(e)[:220]}", flush=True)

print("\n===== I) 2-D conical energy conservation (lossless) =====", flush=True)
for form in ("laurent", "li", "fff_nv"):
    for thd, phid in ((0.0, 0.0), (25.0, 35.0), (50.0, 70.0)):
        for pol in ("te", "tm"):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    r = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell_disk, 1.5, 1.0,
                                           0.3e-6, 0.633e-6,
                                           theta=np.deg2rad(thd),
                                           phi=np.deg2rad(phid),
                                           polarization=pol, n_orders_x=6,
                                           n_orders_y=6, formulation=form)
                Rr, Tt = np.asarray(r[1]), np.asarray(r[2])
                wmsg = [str(x.message)[:55] for x in w]
                print(f"   {form:8s} th={thd:4.1f} phi={phid:4.1f} {pol}: "
                      f"closure={Rr.sum()+Tt.sum()-1:+.3e} warn={wmsg}",
                      flush=True)
            except Exception as e:
                print(f"   {form} th={thd} {pol} RAISED {type(e).__name__}: "
                      f"{str(e)[:160]}", flush=True)

print("\n===== J) 2-D symmetry fold on/off equivalence =====", flush=True)
for sym in (True, False):
    r = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell_disk, 1.5, 1.0, 0.3e-6,
                           0.633e-6, theta=0.0, phi=0.0, polarization="te",
                           n_orders_x=6, n_orders_y=6, formulation="laurent",
                           symmetry=sym)
    print(f"   symmetry={sym}: R00={np.asarray(r[1]).max():.14f} "
          f"closure={np.asarray(r[1]).sum()+np.asarray(r[2]).sum()-1:+.3e}",
          flush=True)

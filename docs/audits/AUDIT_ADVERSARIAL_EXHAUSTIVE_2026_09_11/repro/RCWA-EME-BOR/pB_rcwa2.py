import sys, numpy as np, warnings, time
sys.path.insert(0, r'docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RCWA-EME-BOR')
t0 = time.perf_counter()
from lumenairy.elements.rcwa import (rcwa_efficiency_1d, rcwa_efficiency_2d,
                                     rcwa_jones_2d, rcwa_jones_1d)
print("import %.1f s" % (time.perf_counter() - t0), flush=True)


def fmt(z):
    return f"{abs(z):.10f}<{np.angle(z):+.9f}"


print("\n===== D2) RECIPROCITY (corrected pairing p=+m) =====", flush=True)
args = dict(period=1.3e-6, n_ridge=2.1, n_groove=1.0, depth=0.45e-6,
            duty_cycle=0.4, wavelength=0.633e-6)
nI, nII = 1.0, 1.62
M = 25
for pol in ('te', 'tm'):
    th = np.deg2rad(18.0)
    o, R, T = rcwa_efficiency_1d(n_substrate=nII, n_superstrate=nI, angle=th,
                                 polarization=pol, n_orders=M, **args)
    wl0 = args['wavelength']; Pp = args['period']
    for m in (-2, -1, 0, 1, 2):
        s_m = (nI*np.sin(th) + m*wl0/Pp)/nII
        if abs(s_m) >= 1:
            continue
        th_m = np.arcsin(s_m)
        o2, R2, T2 = rcwa_efficiency_1d(n_substrate=nI, n_superstrate=nII,
                                        angle=-th_m, polarization=pol,
                                        n_orders=M, **args)
        print(f"   T pol={pol} m={m:+d}: fwd={T[M+m]:.12f} rev={T2[M+m]:.12f} "
              f"rel={abs(T[M+m]-T2[M+m])/max(T[M+m],1e-30):.3e}", flush=True)
    # reflection reciprocity (same medium both sides of the pairing)
    for m in (-1, 1):
        s_m = (nI*np.sin(th) + m*wl0/Pp)/nI
        if abs(s_m) >= 1:
            continue
        th_m = np.arcsin(s_m)
        o3, R3, T3 = rcwa_efficiency_1d(n_substrate=nII, n_superstrate=nI,
                                        angle=-th_m, polarization=pol,
                                        n_orders=M, **args)
        print(f"   R pol={pol} m={m:+d}: fwd={R[M+m]:.12f} rev={R3[M+m]:.12f} "
              f"rel={abs(R[M+m]-R3[M+m])/max(R[M+m],1e-30):.3e}", flush=True)

print("\n===== E) grazing-order nudge driven by ANGLE (Rayleigh in theta) =====",
      flush=True)
# lam/P = 0.6, incident from n=1: order -1 grazes when sin(th) - 0.6 = -1 -> th=-23.578 deg
P = 1.0e-6; wl = 0.6e-6
th_c = np.arcsin(-1.0 + wl/P)      # = arcsin(-0.4)
print(f"   critical theta = {np.rad2deg(th_c):.6f} deg", flush=True)
for dth in (-1e-4, -1e-5, -1e-6, -1e-7, 0.0, 1e-7, 1e-6, 1e-5, 1e-4):
    o, R, T = rcwa_efficiency_1d(P, 2.0, 1.0, 1.5, 1.0, 0.4e-6, 0.5, wl,
                                 angle=th_c+dth, polarization='te', n_orders=21)
    print(f"   dtheta={dth:+.1e}  R0={R[21]:.12f}  sum={R.sum()+T.sum():.12f}",
          flush=True)

print("\n===== F) 2-D symmetry tests =====", flush=True)
Sx = 64
xg = (np.arange(Sx))/Sx
X, Y = np.meshgrid(xg, xg, indexing='ij')
# C4-symmetric square pillar centred in the cell
cell_sq = np.where((np.abs(X-0.5) < 0.15) & (np.abs(Y-0.5) < 0.15), 6.25, 2.25)
# centred disk (C-inf)
cell_disk = np.where(((X-0.5)**2 + (Y-0.5)**2) < 0.15**2, 6.25, 2.25)
for name, cell in (("square-C4", cell_sq), ("disk", cell_disk)):
    for form in ("laurent", "li"):
        try:
            out = rcwa_jones_2d(0.5e-6, cell, 1.5, 1.0, 0.3e-6, 0.633e-6,
                                theta=0.0, phi=0.0, n_orders_x=6, n_orders_y=6,
                                formulation=form)
            J = out[3]
            print(f"   {name} {form}: Jxx={fmt(J[0,0])} Jyy={fmt(J[1,1])} "
                  f"|Jxx-Jyy|={abs(J[0,0]-J[1,1]):.3e} |Jxy|={abs(J[0,1]):.3e} "
                  f"|Jyx|={abs(J[1,0]):.3e}", flush=True)
        except Exception as e:
            print(f"   {name} {form} RAISED {type(e).__name__}: {str(e)[:200]}",
                  flush=True)

print("\n===== G) phi-rotation covariance (conical Jones basis, CONVENTIONS 7.1) =====",
      flush=True)
th = np.deg2rad(20.0)
for phid in (0.0, 30.0, 90.0):
    try:
        out = rcwa_jones_2d(0.5e-6, cell_disk, 1.5, 1.0, 0.3e-6, 0.633e-6,
                            theta=th, phi=np.deg2rad(phid), n_orders_x=6,
                            n_orders_y=6, formulation="li")
        J = out[3]
        print(f"   phi={phid:5.1f}: J=[[{fmt(J[0,0])},{fmt(J[0,1])}],"
              f"[{fmt(J[1,0])},{fmt(J[1,1])}]]", flush=True)
        if phid == 0.0:
            J0 = J.copy()
        else:
            c, s = np.cos(np.deg2rad(phid)), np.sin(np.deg2rad(phid))
            Rm = np.array([[c, -s], [s, c]])
            Jpred = Rm @ J0 @ Rm.T
            print(f"      rot-pred residual (C-inf cell) = "
                  f"{np.max(np.abs(J-Jpred)):.3e}", flush=True)
    except Exception as e:
        print(f"   phi={phid} RAISED {type(e).__name__}: {str(e)[:200]}", flush=True)

print("\n===== H) 2-D separable limit vs 1-D solver =====", flush=True)
duty = 0.4
prof = np.where(xg < duty, 4.0, 1.0)              # 1-D in x, uniform in y
cell_1d = np.repeat(prof[:, None], Sx, axis=1)
for pol, row in (("tm", 0), ("te", 1)):
    for ang in (0.0, 30.0):
        o1, R1, T1 = rcwa_efficiency_1d(0.5e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6,
                                        duty, 0.633e-6, angle=np.deg2rad(ang),
                                        polarization=pol, n_orders=8,
                                        formulation='li')
        try:
            out = rcwa_jones_2d(0.5e-6, cell_1d, 1.5, 1.0, 0.3e-6, 0.633e-6,
                                theta=np.deg2rad(ang), phi=0.0, n_orders_x=8,
                                n_orders_y=0, formulation="li")
            o2, R2, T2 = out[0], out[1], out[2]
            m1 = np.array([oo[0] for oo in np.asarray(o2)])
            sel = {int(v): i for i, v in enumerate(m1)}
            dR = max(abs(R1[8+m] - R2[row][sel[m]]) for m in (-1, 0, 1) if m in sel)
            dT = max(abs(T1[8+m] - T2[row][sel[m]]) for m in (-1, 0, 1) if m in sel)
            print(f"   pol={pol} ang={ang}: max|dR|={dR:.3e} max|dT|={dT:.3e} "
                  f"closure2d={R2[row].sum()+T2[row].sum()-1:+.3e}", flush=True)
        except Exception as e:
            print(f"   pol={pol} ang={ang} RAISED {type(e).__name__}: "
                  f"{str(e)[:250]}", flush=True)

print("\n===== I) 2-D conical energy conservation =====", flush=True)
for form in ("laurent", "li", "fff_nv"):
    for thd, phid in ((0.0, 0.0), (25.0, 35.0), (50.0, 70.0)):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                r = rcwa_efficiency_2d(0.5e-6, cell_disk, 1.5, 1.0, 0.3e-6,
                                       0.633e-6, theta=np.deg2rad(thd),
                                       phi=np.deg2rad(phid), n_orders_x=6,
                                       n_orders_y=6, formulation=form)
            Rr, Tt = r[1], r[2]
            print(f"   {form:8s} th={thd:4.1f} phi={phid:4.1f}: "
                  f"closure={np.atleast_2d(Rr).sum(-1)+np.atleast_2d(Tt).sum(-1)-1}"
                  f" warn={[str(x.message)[:60] for x in w]}", flush=True)
        except Exception as e:
            print(f"   {form} th={thd} RAISED {type(e).__name__}: {str(e)[:160]}",
                  flush=True)

import numpy as np, warnings, time
t0 = time.perf_counter()
from lumenairy.elements.rcwa import rcwa_efficiency_1d, rcwa_efficiency_2d
print("import %.1f s" % (time.perf_counter() - t0), flush=True)

print("=== V) 2-D(n_orders_y=0) vs 1-D: convergence in CELL PIXELS Sx ===",
      flush=True)
duty = 0.5
for Sx in (32, 64, 128, 256, 512, 1024, 2048):
    xg = np.arange(Sx)/Sx
    prof = np.where(xg < duty, 4.0, 1.0)
    cell = prof[:, None]
    out = []
    for pol in ("te", "tm"):
        o1, R1, T1 = rcwa_efficiency_1d(0.5e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6,
                                        duty, 0.633e-6, angle=0.0,
                                        polarization=pol, n_orders=8,
                                        formulation='li')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.3e-6,
                                   0.633e-6, theta=0.0, phi=0.0,
                                   polarization=pol, n_orders_x=8,
                                   n_orders_y=0, formulation='li')
        oo, R2, T2 = np.asarray(r[0]), np.asarray(r[1]), np.asarray(r[2])
        sel = {int(v[0]): i for i, v in enumerate(oo)}
        dR = max(abs(R1[8+m] - R2[sel[m]]) for m in (-1, 0, 1) if m in sel)
        dT = max(abs(T1[8+m] - T2[sel[m]]) for m in (-1, 0, 1) if m in sel)
        out.append((pol, dR, dT))
    print("   Sx=%5d  " % Sx +
          "  ".join(f"{p}: dR={a:.2e} dT={b:.2e}" for p, a, b in out),
          flush=True)

print("\n=== W) does eps_cell midpoint sampling change it? ===", flush=True)
for Sx in (64, 256, 1024):
    xg = (np.arange(Sx)+0.5)/Sx
    prof = np.where(xg < duty, 4.0, 1.0)
    cell = prof[:, None]
    o1, R1, T1 = rcwa_efficiency_1d(0.5e-6, 2.0, 1.0, 1.5, 1.0, 0.3e-6, duty,
                                    0.633e-6, angle=0.0, polarization='te',
                                    n_orders=8, formulation='li')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = rcwa_efficiency_2d(0.5e-6, 0.5e-6, cell, 1.5, 1.0, 0.3e-6,
                               0.633e-6, theta=0.0, phi=0.0, polarization='te',
                               n_orders_x=8, n_orders_y=0, formulation='li')
    oo, R2, T2 = np.asarray(r[0]), np.asarray(r[1]), np.asarray(r[2])
    sel = {int(v[0]): i for i, v in enumerate(oo)}
    dR = max(abs(R1[8+m] - R2[sel[m]]) for m in (-1, 0, 1) if m in sel)
    print(f"   midpoint Sx={Sx}: te dR={dR:.3e}", flush=True)

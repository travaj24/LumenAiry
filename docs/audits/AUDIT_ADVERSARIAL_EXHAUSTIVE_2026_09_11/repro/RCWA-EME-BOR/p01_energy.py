import numpy as np, warnings, time
import lumenairy as lm
from lumenairy.elements.rcwa import rcwa_efficiency_1d

print("lumenairy", lm.__version__)
wl=0.633e-6; P=1.0e-6; d=0.4e-6
for pol in ("te","tm"):
    for ang in (0.0, 30.0, 60.0):
        for nsub in (1.0, 1.5):
            o,R,T = rcwa_efficiency_1d(P, 2.0, 1.0, nsub, 1.0, d, 0.5, wl,
                                       angle=np.deg2rad(ang), polarization=pol,
                                       n_orders=31)
            s=R.sum()+T.sum()
            print(f"pol={pol} ang={ang:4.1f} nsub={nsub} sumR={R.sum():.15f} sumT={T.sum():.15f} closure={s-1:+.3e}")

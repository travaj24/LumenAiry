from _fix import BASE, clusters, omega2_numpy
from _v import np

from lumenairy.elements.rcwa import rcwa_efficiency_2d

for n in (2, 3, 4, 5):
    o = np.asarray(rcwa_efficiency_2d(0.9, 0.9, BASE.astype(complex), 1.52, 1.0, 0.37, 1.0, n_orders_x=n, n_orders_y=n, symmetry=False)[0])
    lam = omega2_numpy(BASE, n)
    c6 = clusters(lam, 1e-6)
    c12 = clusters(lam, 1e-12)
    s = np.max(np.abs(lam))
    print(n, "orders", len(o), "eigs", lam.size, "max|lam|", s,
          "clusters(1e-6) sizes", sorted(set(len(c) for c in c6)), len(c6),
          "exact(1e-12)", len(c12), "members", sum(len(c) for c in c6),
          "n_prop", int(np.sum(lam.real < 0)))

"""Probe 4 -- G2 ambiguity search: which reading of Granet's Fig.4 / Eq.37
example (axis assignment, which tensor is the PILLAR, conjugation) reproduces
Table 2, and under which transmitted-efficiency definition.

Reports the four Table-2 orders for every combination; nothing is tuned.
"""
import itertools

import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

LAM = 1.0e-6
EPS_P = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0],
                  [0.0, 0.0, 2.0]], dtype=complex)   # paper eps_b
KEYS = [(1, 1), (-1, 1), (0, -1), (0, 0)]
TAB2 = {(1, 1): 0.0268, (-1, 1): 0.0139, (0, -1): 0.0620, (0, 0): 0.2979}


def run(dx, dy, e_pillar, e_host, n_sub, M=7, pol_row=0):
    cell = np.empty((2, 2, 3, 3), dtype=complex)
    cell[:] = e_host
    cell[0, 0] = e_pillar
    st = PMM2DStackPure(dx * LAM, dy * LAM, n_superstrate=1.0,
                        n_substrate=n_sub, n_modes=M, n_orders=4)
    st.add_layer(LAM, eps_cell=cell)
    st.set_source(LAM, theta=0.0, phi=0.0)
    o, R, T, _J = st.solve()
    a = st.per_order_amplitudes("transmission")
    kx, ky, kz, kzi = a["kx"], a["ky"], a["kz"], a["kz_inc"]
    tx, ty = a["Ex"][pol_row], a["Ey"][pol_row]
    safe = np.where(np.abs(kz) < 1e-12, 1.0, kz)
    tz = -(kx * tx + ky * ty) / safe
    p2 = np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2
    idx = {tuple(int(v) for v in r): i for i, r in enumerate(o)}
    return (idx,
            {"flux": np.real(kz / kzi) * p2,
             "abs2": p2,
             "tang": np.real(kz / kzi) * (np.abs(tx) ** 2 + np.abs(ty) ** 2)})


def main():
    combos = []
    for (dxdy, pillar_is_b, cj, nsub_sign) in itertools.product(
            ((2.4, 1.4), (1.4, 2.4)), (True, False), (True, False),
            (+1, -1)):
        dx, dy = dxdy
        eb = np.conj(EPS_P) if cj else EPS_P
        ea = np.conj(eb)
        e_p, e_h = (eb, ea) if pillar_is_b else (ea, eb)
        n_sub = np.sqrt(1.0 + nsub_sign * 5.0j)
        if np.imag(n_sub) < 0:
            n_sub = -n_sub
        combos.append((f"d=({dx},{dy}) pillar={'b' if pillar_is_b else 'a'} "
                       f"conj={cj} eps_sub=1{'+' if nsub_sign > 0 else '-'}5i",
                       dx, dy, e_p, e_h, n_sub))
    best = []
    for label, dx, dy, e_p, e_h, n_sub in combos:
        idx, defs = run(dx, dy, e_p, e_h, n_sub)
        for dname, arr in defs.items():
            vals = {k: float(np.real(arr[idx[k]])) for k in KEYS}
            dev = max(abs(vals[k] - TAB2[k]) for k in KEYS)
            best.append((dev, label, dname, vals))
    best.sort(key=lambda r: r[0])
    print(f"{'maxdev':>9s}  {'definition':10s} reading")
    for dev, label, dname, vals in best[:12]:
        print(f"{dev:9.2e}  {dname:10s} {label}\n"
              f"           " + "  ".join(f"{str(k)}={vals[k]:.5f}"
                                         for k in KEYS))


if __name__ == "__main__":
    main()

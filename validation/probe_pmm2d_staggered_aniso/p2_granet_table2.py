"""Probe 2 -- G2: Granet 2023 Table 2/3 anisotropic crossed grating.

Resolves BY MEASUREMENT (never by tuning) the two ambiguities the plan flags:
  (i)  the transmitted-efficiency DEFINITION into a LOSSY substrate;
  (ii) the (m, n) order / period-axis convention (the text's "d_y = 2.4,
       d_y = 1.4" is a typo, and the gyrotropic +/- asymmetry is sign-sensitive).

Paper (exp(+i w t)) -> library (exp(-i w t)): every tensor is CONJUGATED.

Run: PYTHONPATH=/c/tmp/lum_aniso python validation/probe_pmm2d_staggered_aniso/p2_granet_table2.py
"""
import itertools

import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402

LAM = 1.0e-6
# paper eps_b (Eq.37) and eps_a = eps_b^*, CONJUGATED into exp(-i w t)
EPS_B_PAPER = np.array([[2.25, -0.5j, 0.0],
                        [0.5j, 2.25, 0.0],
                        [0.0, 0.0, 2.0]], dtype=complex)
EPS_B = np.conj(EPS_B_PAPER)
EPS_A = np.conj(np.conj(EPS_B_PAPER))       # eps_a = eps_b^* (paper) -> conj
N_SUB = np.sqrt(1.0 + 5.0j)                 # paper substrate 1 - 5i, conjugated
# Table 2 (SEM, M>=4) and Table 3 (FMM, M=10) transmitted efficiencies
TAB2_SEM = {(1, 1): 0.0268, (-1, 1): 0.0139, (0, -1): 0.0620, (0, 0): 0.2979}
TAB3_FMM = {(1, 1): 0.0269, (-1, 1): 0.0137, (0, -1): 0.0619, (0, 0): 0.2980}


def build(M, dx_first=True, pillar_at=(0, 0)):
    dx, dy = (2.4, 1.4) if dx_first else (1.4, 2.4)
    cell = np.empty((2, 2, 3, 3), dtype=complex)
    cell[:] = EPS_A
    cell[pillar_at[0], pillar_at[1]] = EPS_B
    st = PMM2DStackPure(dx * LAM, dy * LAM, n_superstrate=1.0,
                        n_substrate=N_SUB, n_modes=M, n_orders=4)
    st.add_layer(LAM, eps_cell=cell)
    st.set_source(LAM, theta=0.0, phi=0.0)
    return st


def variants(st):
    """Return {name: {order: efficiency}} for each candidate T definition."""
    o, R, T, _J = st.solve()
    a = st.per_order_amplitudes("transmission")
    kx, ky, kz, kzi = a["kx"], a["ky"], a["kz"], a["kz_inc"]
    tx, ty = a["Ex"][0], a["Ey"][0]          # incident E_x  (row 0)
    safe = np.where(np.abs(kz) < 1e-12, 1.0, kz)
    tz = -(kx * tx + ky * ty) / safe
    defs = {
        "A_full_Re(kz)|E|^2 (shipped)":
            np.real(kz / kzi) * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                 + np.abs(tz) ** 2),
        "B_tangential_only":
            np.real(kz / kzi) * (np.abs(tx) ** 2 + np.abs(ty) ** 2),
        "C_|kz|-weighted_full":
            np.abs(kz) / kzi * (np.abs(tx) ** 2 + np.abs(ty) ** 2
                                + np.abs(tz) ** 2),
        "D_n_sub_scaled_full":
            np.real(kz / kzi) * np.real(np.sqrt(N_SUB ** 2))
            * (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2),
    }
    idx = {tuple(int(v) for v in row): i for i, row in enumerate(o)}
    out = {"shipped_T_row0": {k: float(T[0][i]) for k, i in idx.items()}}
    for name, arr in defs.items():
        out[name] = {k: float(np.real(arr[i])) for k, i in idx.items()}
    return out


def main():
    keys = [(1, 1), (-1, 1), (0, -1), (0, 0)]
    for dx_first, M in itertools.product((True, False), (5, 7)):
        st = build(M, dx_first=dx_first)
        res = variants(st)
        tag = "dx=2.4,dy=1.4" if dx_first else "dx=1.4,dy=2.4"
        print(f"\n=== {tag}   M={M} ===")
        print(f"{'definition':34s} " + "  ".join(f"{str(k):>9s}" for k in keys)
              + "   maxdev(SEM)  maxdev(FMM)   maxdev(mirrored SEM)")
        for name, d in res.items():
            row = [d[k] for k in keys]
            dev_s = max(abs(d[k] - TAB2_SEM[k]) for k in keys)
            dev_f = max(abs(d[k] - TAB3_FMM[k]) for k in keys)
            mir = {(1, 1): (-1, -1), (-1, 1): (1, -1), (0, -1): (0, 1),
                   (0, 0): (0, 0)}
            dev_m = max(abs(d[mir[k]] - TAB2_SEM[k]) for k in keys)
            print(f"{name:34s} " + "  ".join(f"{v:9.5f}" for v in row)
                  + f"   {dev_s:9.2e}  {dev_f:9.2e}   {dev_m:9.2e}")


if __name__ == "__main__":
    main()

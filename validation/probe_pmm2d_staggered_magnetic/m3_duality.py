"""Probe 3 (G3) -- ELECTROMAGNETIC DUALITY, the exact physics oracle for the
magnetic path that needs no external engine.

Maxwell in the module's PUBLIC ``exp(-i w t)`` gauge,

    curl E = +i w mu0 mu H ,   curl H = -i w eps0 eps E ,

is invariant under  (E, H, eps, mu) -> (Z0 H, -E/Z0, mu, eps).  With VACUUM
half-spaces (self-dual) the grating (eps_cell, mu_cell) and the SWAPPED grating
(mu_cell, eps_cell) therefore solve the same problem, with the incident and
outgoing polarizations rotated in the (s, p) frame:

    incident   s -> p ,  p -> -s
    outgoing   the same rotation D = [[0, -1], [1, 0]] on (a_s, a_p)

so, PER ORDER,

    R_dual(p drive) = R_orig(s drive) ,  R_dual(s drive) = R_orig(p drive)
    (likewise T)                                                       [rule 1]
    J_dual = -D J_orig D^-1       in the (s, p) basis                  [rule 2]
    (the sign is the two p-hat conventions in play -- see ``duality`` below)

The (s, p) frame is used precisely because the map has UNIT magnitude there at
ANY incidence -- in the lab (x, y) tangential basis the same map carries
1/cos(theta) factors and mixes the two drives at conical incidence.

Rule 1 with ``mu_cell = 1`` on one arm is the strongest statement available:
the ELECTRIC arm is the SHIPPED, already-gated anisotropic path, so the
magnetic assembly is being measured against a verified one.

The staggered DISCRETIZATION is not self-dual (E3 is expanded in V3 while H3
lands in Vw), so the two arms agree only to DISCRETIZATION accuracy -- hence
the M ladder: the claim is two-sided (the mismatch SHRINKS with M, and the
M = 8 mismatch sits under a derived bar).
"""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_mag"), \
    lumenairy.__file__

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

P, WL, DEP = 0.70e-6, 0.55e-6, 0.28e-6
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
LC2 = uniaxial_tensor(1.15, 1.35, np.pi / 2, phi=-0.30)
GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                dtype=complex)
EYE = np.eye(3, dtype=complex)


def cell(host, incl, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = incl
    return c


def run(eps_c, mu_c, theta, phi, M):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.0, n_modes=M,
                        n_orders=3)
    if mu_c is None:
        st.add_layer(DEP, eps_cell=eps_c)
    else:
        st.add_layer(DEP, eps_cell=eps_c, mu_cell=mu_c)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve(jones=True)
    ref = st.per_order_amplitudes("reflection")
    trn = st.per_order_amplitudes("transmission")
    return o, ref, trn


def _sp_drives(theta, phi):
    """Tangential (ex0, ey0) of the unit s- and p-polarized incident fields."""
    s = (-np.sin(phi), np.cos(phi))
    p = (np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi))
    return s, p


def eff(amps, drive, kz_inc, kx0, ky0):
    """Per-order efficiency for an arbitrary tangential drive -- the library's
    own projection (rcwa._core._project_efficiency) applied to the linear
    combination of the two returned drive rows."""
    ex0, ey0 = drive
    Ex = ex0 * amps["Ex"][0] + ey0 * amps["Ex"][1]
    Ey = ex0 * amps["Ey"][0] + ey0 * amps["Ey"][1]
    kx, ky, kz = amps["kx"], amps["ky"], amps["kz"]
    safe = np.where(np.abs(kz) < 1e-12, 1.0, kz)
    Ez = -(kx * Ex + ky * Ey) / safe
    long_inc = kx0 * ex0 + ky0 * ey0
    # |E_inc|^2 = |e_t|^2 + |e_z|^2.  The library hardcodes the leading 1.0
    # because ITS two drives are unit-TANGENTIAL; an (s, p) drive is unit in
    # the FULL field, so the tangential norm must be carried explicitly (with
    # 1.0 there the p arm mis-normalises by |e_t|^2 = cos^2 theta -- MEASURED
    # 0.0142 vs the analytic 0.0159 at theta = 0.35).
    einc_sq = (abs(ex0) ** 2 + abs(ey0) ** 2)
    if kz_inc != 0:
        einc_sq = einc_sq + (long_inc / kz_inc) ** 2
    e = (np.real(kz / kz_inc)
         * (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2) / einc_sq)
    return np.where(np.real(kz) > 0, np.real(e), 0.0)


def jones_sp(amps, o, theta, phi):
    """Order-0 REFLECTION Jones in the (s, p) basis, columns = (s, p) drives."""
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    sd, pd = _sp_drives(theta, phi)
    ct, sp_, cp = np.cos(theta), np.sin(phi), np.cos(phi)
    out = np.zeros((2, 2), dtype=complex)
    for col, drive in enumerate((sd, pd)):
        Ex = drive[0] * amps["Ex"][0][p0] + drive[1] * amps["Ex"][1][p0]
        Ey = drive[0] * amps["Ey"][0][p0] + drive[1] * amps["Ey"][1][p0]
        out[0, col] = -sp_ * Ex + cp * Ey                    # a_s
        out[1, col] = (cp * Ex + sp_ * Ey) / ct              # a_p
    return out


D = np.array([[0.0, -1.0], [1.0, 0.0]])


def duality(tag, eps_c, mu_c, theta, phi, M):
    o, refA, trnA = run(eps_c, mu_c if mu_c is not None else None,
                        theta, phi, M)
    mu_swap = eps_c
    eps_swap = mu_c if mu_c is not None else np.broadcast_to(
        EYE, eps_c.shape).copy()
    o2, refB, trnB = run(eps_swap, mu_swap, theta, phi, M)
    assert np.array_equal(o, o2)
    sd, pd = _sp_drives(theta, phi)
    kzi, kx0, ky0 = refA["kz_inc"], refA["kx0"], refA["ky0"]
    d = 0.0
    for port, A, B in (("R", refA, refB), ("T", trnA, trnB)):
        for drA, drB in ((sd, pd), (pd, sd)):
            d = max(d, float(np.max(np.abs(
                eff(A, drA, kzi, kx0, ky0) - eff(B, drB, kzi, kx0, ky0)))))
    JA = jones_sp(refA, o, theta, phi)
    JB = jones_sp(refB, o, theta, phi)
    # rule 2, with the sign that the two p-hat conventions in play imply:
    # the INCIDENT p-hat here is the optics convention
    # (cos th cos ph, cos th sin ph, -sin th) = -(k^ x s^), while the
    # REFLECTED p-hat is +(k^_r x s^); that one flip turns D J D^-1 into
    # -D J D^-1 (MEASURED: 0.32 with the unflipped sign, 1e-14 with this one).
    dJ = float(np.max(np.abs(JB + D @ JA @ np.linalg.inv(D))))
    # negative control: the UNROTATED comparison must be visibly worse
    dJ0 = float(np.max(np.abs(JB - JA)))
    print(f"{tag:<46s} th={theta:.2f} ph={phi:.2f} M={M}  "
          f"dRT={d:.3e}  dJ_sp={dJ:.3e}  (no-rotation control {dJ0:.2e})")
    return d, dJ, dJ0


if __name__ == "__main__":
    print("=== A. PATTERNED electric cell vs its PURELY MAGNETIC dual "
          "(mu = 1 on arm A: the shipped anisotropic path is the oracle)")
    ec = cell(LC, 4.0 * EYE)
    for th, ph in ((0.0, 0.0), (0.30, 0.0), (0.30, 0.7)):
        for M in (5, 6, 7, 8):
            duality("eps = LC host + iso pillar, mu = 1", ec, None, th, ph, M)
    print()
    print("=== B. UNIFORM (eps, mu) TENSOR pair -- expected SPECTRAL")
    ue = np.broadcast_to(LC, (2, 2, 3, 3)).copy()
    um = np.broadcast_to(LC2, (2, 2, 3, 3)).copy()
    for th, ph in ((0.0, 0.0), (0.30, 0.7)):
        for M in (5, 6, 7, 8):
            duality("uniform eps=LC, mu=LC2", ue, um, th, ph, M)
    print()
    print("=== C. PATTERNED (eps, mu) both anisotropic, both patterned")
    ec2 = cell(LC, GYRO)
    mc2 = cell(LC2, 1.6 * EYE)
    for th, ph in ((0.0, 0.0), (0.30, 0.7)):
        for M in (5, 6, 7, 8):
            duality("eps = LC/GYRO cell, mu = LC2/1.6 cell", ec2, mc2,
                    th, ph, M)

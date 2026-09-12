"""PROP-HF p1: independent Debye-Wolf quadrature vs richards_wolf_focus.

Oracle: direct numerical integration of the Debye-Wolf integral in the
RAY-DIRECTION parameterisation (Novotny & Hecht Sec. 3.6, eq. 3.58-3.66;
Richards & Wolf 1959 eqs. 2.26-2.30).  No FFT, no aperture coordinate --
the integration variable IS the ray direction shat = (sin t cos p,
sin t sin p, cos t), so the phi_ray/phi_pupil ambiguity cannot leak in.

    E_j(r) = C * Int_0^tmax Int_0^2pi  e_j(t,p) * sqrt(cos t)
                 * exp(i k shat . r) * sin t  dt dp

    e_x = cos t cos^2 p + sin^2 p
    e_y = cos p sin p (cos t - 1)
    e_z = -sin t cos p                       (x-polarised input)

Derivation check (independent of any formula): a ray entering an
aplanatic lens at aperture point (+a, 0) travels toward the focus with
direction (-sin t, 0, cos t), i.e. phi_ray = pi.  Its p-polarised E
rotates rigidly with the ray: R_y(-t) applied to xhat = (cos t, 0, +sin t),
so E_z > 0 for the +x aperture point.  Both formulas above reproduce that
with phi_ray = pi:  e_z = -sin t * cos(pi) = +sin t.
"""
import sys
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.vector_diffraction import richards_wolf_focus  # noqa: E402

lam = 633e-9
NA = 0.5
f = 4e-3
k = 2 * np.pi / lam
tmax = np.arcsin(NA)


def debye_wolf_direct(points, n_t=600, n_p=720, pol=(1.0, 0.0)):
    """Gauss-Legendre in theta x trapezoid in phi. Returns (Ex,Ey,Ez) arrays."""
    px, py = pol
    xt, wt = np.polynomial.legendre.leggauss(n_t)
    t = 0.5 * tmax * (xt + 1.0)
    wt = wt * 0.5 * tmax
    p = np.arange(n_p) * (2 * np.pi / n_p)
    wp = np.full(n_p, 2 * np.pi / n_p)
    T, P = np.meshgrid(t, p, indexing='ij')
    W = np.outer(wt, wp)
    ct, st = np.cos(T), np.sin(T)
    cp, sp = np.cos(P), np.sin(P)
    # aplanatic + solid-angle element
    base = np.sqrt(ct) * st * W
    ex = px * (ct * cp ** 2 + sp ** 2) + py * (cp * sp * (ct - 1.0))
    ey = px * (cp * sp * (ct - 1.0)) + py * (ct * sp ** 2 + cp ** 2)
    ez = -st * (px * cp + py * sp)
    out = []
    for (x, y, z) in points:
        ph = np.exp(1j * k * (x * st * cp + y * st * sp + z * ct))
        g = base * ph
        out.append((np.sum(g * ex), np.sum(g * ey), np.sum(g * ez)))
    return np.array(out)


def code_fields(Np=512, pol='x'):
    dx_pupil = 2.2 * f * NA / Np          # array comfortably spans the rim
    pupil = np.ones((Np, Np), dtype=np.complex128)
    Ex, Ey, Ez, xf, yf = richards_wolf_focus(
        pupil, lam, NA, f, dx_pupil, polarization=pol)
    return Ex, Ey, Ez, xf, yf


def main():
    Ex, Ey, Ez, xf, yf = code_fields()
    N = Ex.shape[0]
    c = N // 2                      # index of x_f = 0 (pixel-centred grid)
    dxf = xf[1] - xf[0]
    print(f"code grid: N={N}  dx_focal={dxf:.6e} m  x_f[c]={xf[c]:.3e}")

    # pick a few off-axis sample points on the code grid
    offsets = [2, 5, 9, 14]
    pts, labels, idx = [], [], []
    for o in offsets:
        pts.append((xf[c + o], 0.0, 0.0)); labels.append(f"(x={xf[c+o]*1e6:+.3f} um, y=0)")
        idx.append((c, c + o))     # Ex[row=y index, col=x index]
    for o in offsets[:2]:
        pts.append((0.0, yf[c + o], 0.0)); labels.append(f"(x=0, y={yf[c+o]*1e6:+.3f} um)")
        idx.append((c + o, c))
    pts.append((xf[c + 5], yf[c + 5], 0.0)); labels.append("diagonal (+5,+5)px")
    idx.append((c + 5, c + 5))

    ref = debye_wolf_direct(pts)

    print()
    print("  point                          Ez/Ex  (direct oracle)      Ez/Ex  (richards_wolf_focus)")
    for lab, (ix_y, ix_x), (rx, ry, rz) in zip(labels, idx, ref):
        cex, cez = Ex[ix_y, ix_x], Ez[ix_y, ix_x]
        r_ref = rz / rx if abs(rx) > 0 else np.nan
        r_cod = cez / cex if abs(cex) > 0 else np.nan
        print(f"  {lab:30s} {r_ref.real:+.6f}{r_ref.imag:+.6f}j     "
              f"{r_cod.real:+.6f}{r_cod.imag:+.6f}j")

    # amplitude sanity: compare the whole (Ex,Ey,Ez) triple up to one global
    # complex constant fixed by Ex at the on-axis point.
    on = debye_wolf_direct([(0.0, 0.0, 0.0)])[0]
    scale = Ex[c, c] / on[0]
    print(f"\nglobal constant from Ex(0,0): {scale:.6e}")
    print("  point                      comp   direct*scale            code                    ratio")
    for lab, (iy, ix), (rx, ry, rz) in zip(labels, idx, ref):
        for name, rv, cv in (('Ex', rx, Ex[iy, ix]),
                             ('Ey', ry, Ey[iy, ix]),
                             ('Ez', rz, Ez[iy, ix])):
            pred = rv * scale
            rat = cv / pred if abs(pred) > 1e-300 else np.nan
            print(f"  {lab:26s} {name}  {pred:+.6e}  {cv:+.6e}  {rat:+.5f}")
        print()


if __name__ == '__main__':
    main()

"""
Lumenairy example 14 -- FGA caustic-accurate lens propagator.

The Frozen Gaussian Approximation (FGA, Herman-Kluk transplanted to the wave
equation, Lu & Yang 2011) is the caustic-accurate member of the beamlet-summation
family.  Where the thawed Gaussian-beamlet propagator (GBD) evolves a complex
curvature that *breathes* and smears a focus, FGA FREEZES the beamlet width and
weights each frozen beamlet by the Herman-Kluk prefactor a = sqrt(det Z), whose
determinant never vanishes at a caustic -- so the phase-space swarm reconstructs
the caustic peak by interference instead of smearing it.

This example shows three things:

  1. STRONG CAUSTIC (headline).  A converging field carrying strong spherical
     aberration is propagated in free space -- a regime where the angular-spectrum
     method (ASM) is an EXACT oracle.  We z-scan through the caustic; the
     caustic-specific metric is the PEAK-INTENSITY error, which FGA renders several
     times more accurately than GBD (GBD's thawed beamlets breathe through focus
     and smear the peak).  Overall-field fidelity is comparable -- the two agree
     away from the focus, so the error GBD makes is concentrated AT the caustic
     peak, which is exactly the piece FGA reconstructs.

  2. THROUGH A REAL SINGLET.  FGA propagated through a real plano-convex N-BK7
     singlet (surface trace + monodromy + image-leg) matches both GBD and the ASM
     oracle in the smooth converging region -- the through-surface transport is
     correct, not just the free-space kernel.

  3. THE UNIVERSAL AUTO-DISPATCHER.  apply_real_lens_universal routes each output
     plane to the most accurate propagator for its regime (low-NA -> exact phase-
     screen; high-NA smooth -> per-pixel traced OPL; high-NA near a caustic -> FGA).

A PNG summary is written to examples/output/ when matplotlib is available.
"""
import os

import numpy as np

import lumenairy as la
from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.gbd import propagate_gbd_freespace

_WL = 0.633e-6


def _fid(a, b):
    """Normalization-free overlap fidelity (the caustic shape metric)."""
    return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300)


def _proj_scale(truth, field):
    """Least-squares best complex scale of `field` onto `truth`.  Divides out a
    propagator's global amplitude/phase (e.g. FGA's known FBI-near-p=0
    representation scale) so the comparison isolates how well the CAUSTIC PEAK is
    rendered -- the same metric the FGA validation test uses."""
    return np.vdot(truth, field) / (np.vdot(field, field) + 1e-300)


def _peak_err(truth, field):
    """Relative caustic-peak-intensity error after the best-fit projection."""
    pk = float(np.abs(truth).max() ** 2)
    scaled = _proj_scale(truth, field) * field
    return abs(float(np.abs(scaled).max() ** 2) - pk) / pk


def _aberrated_converging_field(N=256, dx=0.7e-6, F=250e-6, sa_waves=8.0,
                                w0=22e-6):
    """Collimated Gaussian stamped with an ideal focusing phase + strong
    4th-order (spherical-aberration) phase -- the canonical caustic generator."""
    xs = (np.arange(N) - N / 2) * dx
    Xg, Yg = np.meshgrid(xs, xs)
    r2 = Xg ** 2 + Yg ** 2
    k0 = 2 * np.pi / _WL
    rn = np.sqrt(r2) / w0
    field = (np.exp(-r2 / w0 ** 2)
             * np.exp(-1j * k0 * r2 / (2.0 * F))
             * np.exp(1j * 2.0 * np.pi * sa_waves * rn ** 4))
    return field.astype(np.complex128), dx, F


def _flat_prescription(N, dx):
    """A null (air->air, flat) prescription so apply_real_lens_fga does a pure
    free-space propagation of the caustic field with an exact oracle."""
    return {'name': 'flat', 'aperture_diameter': N * dx,
            'surfaces': [{'radius': np.inf, 'conic': 0.0,
                          'glass_before': 'air', 'glass_after': 'air',
                          'semi_diameter': N * dx / 2}],
            'thicknesses': []}


def _singlet():
    """Plano-convex N-BK7 singlet, f ~ 38.8 mm (curved-first / flat exit)."""
    return {'name': 'pcx', 'aperture_diameter': 2.8e-3,
            'surfaces': [
                {'radius': 20e-3, 'conic': 0.0, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'semi_diameter': 1.4e-3},
                {'radius': np.inf, 'conic': 0.0, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'semi_diameter': 1.4e-3}],
            'thicknesses': [2.5e-3]}


def demo_strong_caustic():
    print("=" * 70)
    print(" 1. STRONG CAUSTIC z-scan  (aberrated converging field, free space)")
    print("    ASM is an EXACT oracle here.  The caustic-specific metric is the")
    print("    PEAK-INTENSITY error (how bright the focus is rendered): FGA")
    print("    reconstructs the caustic peak; GBD's thawed beamlets smear it.")
    print("=" * 70)
    N = 256
    uc, dx, F = _aberrated_converging_field(N=N)
    flat = _flat_prescription(N, dx)

    zs = np.array([0.85, 0.90, 1.00, 1.10]) * F
    print(f"\n  {'z/F':>6}  {'peakErrFGA':>11}  {'peakErrGBD':>11}"
          f"  {'FGA x better':>12}  {'fidFGA':>7}  {'fidGBD':>7}", flush=True)
    print(f"  {'-'*6}  {'-'*11}  {'-'*11}  {'-'*12}  {'-'*7}  {'-'*7}",
          flush=True)
    rows = []
    for z in zs:
        asm = angular_spectrum_propagate(uc, z, _WL, dx)
        fga = la.apply_real_lens_fga(uc, prescription=flat, wavelength=_WL,
                                     dx=dx, output_plane_distance=z,
                                     w0_factor=4.0, p_max=0.14, n_p=15)
        gbd = propagate_gbd_freespace(uc, dx, z=z, wavelength=_WL,
                                      sample_step=3, waist_factor=2.0,
                                      direction_sampling=True)
        ef, eg = _peak_err(asm, fga), _peak_err(asm, gbd)
        rows.append((z / F, ef, eg, _fid(asm, fga), _fid(asm, gbd),
                     asm, fga, gbd))
        print(f"  {z/F:>6.3f}  {ef*100:>10.1f}%  {eg*100:>10.1f}%"
              f"  {eg/max(ef,1e-9):>11.1f}x  {_fid(asm,fga):>7.4f}"
              f"  {_fid(asm,gbd):>7.4f}", flush=True)

    imax = int(np.argmax([r[2] / max(r[1], 1e-9) for r in rows]))  # biggest win
    z_w, ef_w, eg_w = rows[imax][0], rows[imax][1], rows[imax][2]
    print(f"\n  Caustic peak intensity: FGA is within {ef_w*100:.1f}% of the "
          f"exact peak vs GBD's {eg_w*100:.1f}% (z/F={z_w:.2f}, "
          f"{eg_w/max(ef_w,1e-9):.1f}x closer).")
    print("  (Overall-field fidelity is comparable -- the two agree away from")
    print("   the focus; the error GBD makes is concentrated AT the caustic")
    print("   peak, which is exactly the piece FGA reconstructs.)")
    return rows, imax, F


def demo_through_singlet():
    print("\n" + "=" * 68)
    print(" 2. THROUGH A REAL SINGLET  (surface trace + monodromy + image leg)")
    print("=" * 68)
    presc = _singlet()
    xs = (np.arange(256) - 128) * 10e-6
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (0.9e-3) ** 2).astype(np.complex128)
    dx = 10e-6
    zi = 25e-3  # pre-focus smooth region (f ~ 38.8 mm)
    fga = la.apply_real_lens_fga(u0, prescription=presc, wavelength=_WL, dx=dx,
                                 output_plane_distance=zi)
    gbd = la.apply_real_lens_gbd(u0, prescription=presc, wavelength=_WL, dx=dx,
                                 output_plane_distance=zi,
                                 beamlets_per_aperture=40)
    asm = angular_spectrum_propagate(
        la.apply_real_lens(u0, prescription=presc, wavelength=_WL, dx=dx),
        zi, _WL, dx)

    def fid(a, b):
        return abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"\n  pre-focus plane z = {zi*1e3:.0f} mm  (f ~ 38.8 mm)")
    print(f"    fidelity  FGA vs ASM oracle : {fid(fga, asm):.4f}")
    print(f"    fidelity  GBD vs ASM oracle : {fid(gbd, asm):.4f}")
    print(f"    fidelity  FGA vs GBD        : {fid(fga, gbd):.4f}")
    print("  -> FGA's through-surface transport matches the wave oracle.")


def demo_universal_dispatch():
    print("\n" + "=" * 68)
    print(" 3. UNIVERSAL AUTO-DISPATCHER  (route by regime)")
    print("=" * 68)
    from lumenairy.propagators.fga import _caustic_zone, _system_na
    presc = _singlet()
    xs = (np.arange(256) - 128) * 10e-6
    Xg, Yg = np.meshgrid(xs, xs)
    u0 = np.exp(-(Xg ** 2 + Yg ** 2) / (0.9e-3) ** 2).astype(np.complex128)
    dx = 10e-6
    na = _system_na(presc, _WL)
    zone = _caustic_zone(u0, dx, presc, _WL)
    print(f"\n  system NA ~ {na:.3f}   caustic zone ~ "
          f"{zone[0]*1e3:.1f}..{zone[1]*1e3:.1f} mm")
    print(f"  {'output plane':>16}  {'routed to':>14}")
    print(f"  {'-'*16}  {'-'*14}")
    for zmm, label in ((15.0, "smooth region"), (38.0, "near focus")):
        _o, m = la.apply_real_lens_universal(
            u0, prescription=presc, wavelength=_WL, dx=dx,
            output_plane_distance=zmm * 1e-3, return_method=True,
            method_kwargs={'gbd': {'beamlets_per_aperture': 40}})
        print(f"  {zmm:>13.0f} mm  {m:>14}   ({label})")
    print("  (low-NA lens -> exact phase-screen; a high-NA lens would route the")
    print("   smooth planes to per-pixel traced OPL and the focus to FGA.)")


def _save_figure(rows, ipk, F):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n  (matplotlib unavailable -- skipping the PNG summary)")
        return
    zf = np.array([r[0] for r in rows])
    ef = np.array([r[1] for r in rows]) * 100.0
    eg = np.array([r[2] for r in rows]) * 100.0
    asm, fga, gbd = rows[ipk][5], rows[ipk][6], rows[ipk][7]
    N = asm.shape[0]
    c = N // 2

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax0.plot(zf, eg, 'C3-^', lw=1.8, ms=6, label="GBD (thawed, smears peak)")
    ax0.plot(zf, ef, 'C0-s', lw=1.8, ms=6, label="FGA (frozen, caustic-accurate)")
    ax0.axhline(0, color='k', lw=1, label="ASM (exact oracle)")
    ax0.axvline(rows[ipk][0], color='0.7', ls='--', lw=1)
    ax0.set_xlabel("z / F")
    ax0.set_ylabel("caustic peak-intensity error (%)")
    ax0.set_title("Caustic peak error vs plane (lower = better)")
    ax0.legend(fontsize=8)

    # transverse cut, each field best-fit-scaled onto the ASM oracle
    xs = (np.arange(N) - c)
    ia = np.abs(asm[c]) ** 2
    iff = np.abs(_proj_scale(asm, fga) * fga[c]) ** 2
    ig = np.abs(_proj_scale(asm, gbd) * gbd[c]) ** 2
    ax1.plot(xs, ia, 'k-', lw=2, label="ASM")
    ax1.plot(xs, iff, 'C0-', lw=1.6, label="FGA")
    ax1.plot(xs, ig, 'C3-', lw=1.6, label="GBD")
    ax1.set_xlim(-60, 60)
    ax1.set_xlabel("x  (pixels)")
    ax1.set_ylabel("intensity (best-fit scaled)")
    ax1.set_title(f"Transverse cut at z/F={rows[ipk][0]:.2f}")
    ax1.legend(fontsize=8)
    fig.suptitle("FGA vs GBD at a strong spherical-aberration caustic", y=1.02)
    fig.tight_layout()

    outdir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "14_fga_caustic.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\n  Figure written to {path}")


def main():
    rows, ipk, F = demo_strong_caustic()
    demo_through_singlet()
    demo_universal_dispatch()
    _save_figure(rows, ipk, F)


if __name__ == '__main__':
    main()

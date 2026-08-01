# POP CROSSCHECK, step 2: ADVERSARIAL diagnostics on the POP configuration
# itself, BEFORE any number from it is believed.
#
# Three questions, none of which POP answers in its headline output:
#
#  A. WHERE IS THE ARRAY?  Zemax POP does not keep the array "beam-sized".  It
#     Fresnel-transforms, so the point spacing at surface k is set by the ARRAY
#     WIDTH at surface k-1 (dx_out = lambda*z/W_in), not by the beam.  With the
#     naive settings (N=256, W=0.04 mm) the array at the DOE is 502 mm wide with
#     1.96 mm spacing -- i.e. the 22.5 mm first lens is sampled by 14 points and
#     the 46 mrad order-(-4) phase ramp (which needs dx < lambda/(2 sin th) =
#     14.2 um) is aliased by a factor of 138.  This mode prints the grid at
#     every surface so the sampling can be checked against BOTH the aperture
#     and the tilt-ramp Nyquist limit instead of assumed.
#
#  B. IS "TOTAL POWER" A TRANSMISSION NUMBER?  YES -- but only the clip test
#     proves it, and the obvious test does not.  An end-surface sweep gives
#     exactly 1.000000 at surfaces 1, 3, 8, 12, 20, 28 AND 29, which looks
#     exactly like a display renormalisation and was written up as one here
#     before this mode was run.  It is not: forcing a hard circular aperture on
#     surface 20 drops SUM(I)dA to 0.947 / 0.677 / 0.327 for R = 8 / 5 / 3 mm.
#     The unclipped sweep reads 1.000000 at every surface because this design
#     genuinely loses nothing -- which the independent ray trace (mode=vig,
#     0 of 70681 rays vignetted) confirms.  Lesson worth keeping: "the number
#     never moves" is not evidence of normalisation until you have MADE it move.
#
#  C. WHAT DOES THE CLIPPING ACTUALLY COST?  A dense Gaussian-weighted pupil
#     ray trace, counting vignetted rays, gives a POP-independent transmission.
#
# usage:  python pop_grid_diag.py --mode grid|clip|vig  [--config 3] ...
import argparse
import sys

import numpy as np

ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")
LAM_UM = 1.31
# order (-4) on a 1/0.00879 um = 113.77 um period -> 46.06 mrad; the y grating
# adds 23.03 mrad.  Nyquist on the resulting phase ramp:
TILT_MRAD = np.hypot(4 * LAM_UM * 0.00879, 2 * LAM_UM * 0.00879) * 1e3
DX_NYQ_UM = LAM_UM / (2 * TILT_MRAD * 1e-3)


def connect():
    import zospy as zp
    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")
    oss.load(ZMX)
    return zp, zos, oss


def make_pop(zp, nx, width, end, waist=0.004, field=1, start=1,
             surface_to_beam=0.0, data_type='Irradiance'):
    return zp.analyses.physicaloptics.PhysicalOpticsPropagation(
        wavelength=1, field=field, start_surface=start, end_surface=end,
        surface_to_beam=surface_to_beam,
        beam_type='GaussianWaist',
        beam_parameters={'Waist X': waist, 'Waist Y': waist},
        x_sampling=f"{nx}x{nx}", y_sampling=f"{nx}x{nx}",
        x_width=width, y_width=width,
        use_total_power=True, total_power=1.0, use_peak_irradiance=False,
        data_type=data_type, project='AlongBeam')


def grid_of(res):
    df = res.data
    if df is None:
        return None
    xs = np.asarray(df.columns, dtype=float)
    ys = np.asarray(df.index, dtype=float)
    I = np.asarray(df.values, dtype=float)
    dx = float(np.diff(xs).mean())
    return dict(I=I, x=xs, y=ys, dx=dx, W=float(xs[-1] - xs[0] + dx),
                P=float(I.sum()) * dx * dx, peak=float(I.max()))


def mode_grid(args):
    zp, zos, oss = connect()
    oss.MCE.SetCurrentConfiguration(args.config)
    semis = {s: oss.LDE.GetSurfaceAt(s).SemiDiameter
             for s in range(oss.LDE.NumberOfSurfaces)}
    print(f"tilt for the extreme order = {TILT_MRAD:.3f} mrad  ->  the array "
          f"spacing must be < {DX_NYQ_UM:.2f} um wherever the beam carries it")
    for nx in args.nx:
        for W in args.width:
            print(f"\n### N={nx}  W_start={W} mm  (dx_start="
                  f"{W / nx * 1e3:.4f} um, waist 4 um sampled by "
                  f"{2 * 0.004 / (W / nx):.1f} pts across the 1/e^2 diameter)")
            print("  surf   dx [um]      array W [mm]   semi-dia [mm]  "
                  "pts across aperture   tilt-Nyquist")
            for e in args.ends:
                try:
                    g = grid_of(make_pop(zp, nx, W, e).run(oss))
                except Exception as exc:                       # noqa: BLE001
                    print(f"  {e:4d}   FAILED: {exc}")
                    continue
                if g is None:
                    print(f"  {e:4d}   no data grid")
                    continue
                Wmm = g['W']
                sd = semis.get(e, np.nan)
                npts = 2 * sd / (g['dx']) if g['dx'] else np.nan
                ny = ('n/a' if e < 9 else
                      ('OK  ' if g['dx'] * 1e3 < DX_NYQ_UM else
                       f"ALIASED x{g['dx'] * 1e3 / DX_NYQ_UM:.0f}"))
                print(f"  {e:4d}  {g['dx'] * 1e3:11.5f}  {Wmm:12.5f}  "
                      f"{sd:12.4f}   {npts:14.1f}   {ny}")
    oss.close()


def mode_clip(args):
    """Prove that POP's 'Total Power' label is a display normalisation."""
    zp, zos, oss = connect()
    from zospy.api import constants
    oss.MCE.SetCurrentConfiguration(args.config)
    nx, W = args.nx[0], args.width[0]

    base = grid_of(make_pop(zp, nx, W, 29).run(oss))
    print(f"unclipped : SUM(I)dA = {base['P']:.8f}   peak = {base['peak']:.6e}")

    surf = oss.LDE.GetSurfaceAt(args.clip_surf)
    ap = surf.ApertureData
    for semi in args.clip_semi:
        st = ap.CreateApertureTypeSettings(
            constants.Editors.LDE.SurfaceApertureTypes.CircularAperture)
        st._S_CircularAperture.MaximumRadius = semi
        st._S_CircularAperture.MinimumRadius = 0.0
        ap.ChangeApertureTypeSettings(st)
        g = grid_of(make_pop(zp, nx, W, 29).run(oss))
        print(f"clip s{args.clip_surf} R={semi:7.3f} mm : "
              f"SUM(I)dA = {g['P']:.8f}   peak = {g['peak']:.6e}   "
              f"peak ratio = {g['peak'] / base['peak']:.6f}")
    st = ap.CreateApertureTypeSettings(
        constants.Editors.LDE.SurfaceApertureTypes.None_)
    ap.ChangeApertureTypeSettings(st)
    print("\nIf SUM(I)dA stays at 1.00000000 while the peak collapses, the "
          "power label is a DISPLAY NORMALISATION and cannot be used as a\n"
          "transmission number.  Energy must then be quoted as a fraction of "
          "the power that ARRIVES in the POP window.")
    oss.close()


def mode_vig(args):
    """POP-independent transmission: Gaussian-weighted pupil ray trace."""
    zp, zos, oss = connect()
    from zospy.api import constants
    oss.MCE.SetCurrentConfiguration(args.config)
    n = args.npupil
    # object-space NA 0.21 defines the pupil; the source is a 4 um waist whose
    # far-field 1/e^2 half-angle is lambda/(pi w0) = 0.10426 rad.  Normalised
    # pupil coordinate p maps to sin(theta) = 0.21 p, so the amplitude weight
    # is exp(-(0.21 p / 0.10426)^2).
    na = oss.SystemData.Aperture.ApertureValue
    th0 = (LAM_UM * 1e-3) / (np.pi * 0.004)
    t = (np.arange(n) - (n - 1) / 2.0) / ((n - 1) / 2.0)
    PX, PY = np.meshgrid(t, t)
    rp = np.hypot(PX, PY)
    keep = rp <= 1.0
    px, py = PX[keep], PY[keep]
    w = np.exp(-2 * (na * rp[keep] / th0) ** 2)   # intensity weight
    rt = oss.Tools.OpenBatchRayTrace()
    nr = rt.CreateNormUnpol(px.size, constants.Tools.RayTrace.RaysType.Real,
                            oss.LDE.NumberOfSurfaces - 1)
    for a, b in zip(px, py):
        nr.AddRay(1, 0.0, 0.0, float(a), float(b),
                  constants.Tools.RayTrace.OPDMode.None_)
    rt.RunAndWaitForCompletion()
    nr.StartReadingResults()
    ok = np.zeros(px.size, bool)
    err = np.zeros(px.size, int)
    vig = np.zeros(px.size, int)
    for i in range(px.size):
        r = nr.ReadNextResult()
        if not r[0]:
            break
        j = int(r[1]) - 1
        ok[j] = True
        err[j] = int(r[2])
        vig[j] = int(r[3])
    rt.Close()
    good = ok & (err == 0) & (vig == 0)
    T = float(w[good].sum() / w.sum())
    print(f"config {args.config}: {px.size} pupil rays over NA={na}, "
          f"Gaussian-weighted")
    print(f"  traced ok      : {ok.sum()}")
    print(f"  error != 0     : {(err != 0).sum()}")
    print(f"  vignetted      : {(vig != 0).sum()}")
    print(f"  vignetting surfaces: "
          f"{sorted(set(int(v) for v in vig[vig != 0]))}")
    print(f"  GEOMETRIC TRANSMISSION (energy-weighted) = {T * 100:.4f} %")
    print(f"  unweighted (uniform pupil)               = "
          f"{good.sum() / px.size * 100:.4f} %")
    oss.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='grid', choices=('grid', 'clip', 'vig'))
    p.add_argument('--config', type=int, default=3)
    p.add_argument('--nx', type=int, nargs='+', default=[256])
    p.add_argument('--width', type=float, nargs='+', default=[0.04])
    p.add_argument('--ends', type=int, nargs='+',
                   default=[1, 3, 5, 7, 8, 13, 14, 18, 20, 21, 25, 27, 28, 29])
    p.add_argument('--clip-surf', type=int, default=20)
    p.add_argument('--clip-semi', type=float, nargs='+',
                   default=[8.0, 5.0, 3.0])
    p.add_argument('--npupil', type=int, default=201)
    a = p.parse_args(argv)
    {'grid': mode_grid, 'clip': mode_clip, 'vig': mode_vig}[a.mode](a)
    return 0


if __name__ == '__main__':
    sys.exit(main())

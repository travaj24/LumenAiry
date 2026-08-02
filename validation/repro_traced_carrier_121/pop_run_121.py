# POP CROSSCHECK, step 1: run Zemax Physical Optics Propagation on design 121
# and dump the raw image-plane irradiance grid to .npz for offline metrics.
#
# WHY A SEPARATE DUMPER.  Every metric argument in this campaign has been lost
# at least once to a harness that recomputed the metric a slightly different way
# from the reference (audit SCOPE_TILTED_COARSE_LEG_TRANSPORT S8).  So this
# script does NOTHING but drive OpticStudio and save I(x, y); the comparison
# script computes FWHM / EE / power from the saved grid using code lifted
# verbatim from our own scorers.
#
# ORDER SELECTION.  The .zmx carries the DOE order in its multi-configuration
# editor, on parameter 2 of the two DGRATING surfaces (9 = deflects in GLOBAL Y,
# 11 = deflects in GLOBAL X because surfaces 10/12 rotate +/-90 deg about z):
#     cfg 1 -> (0, 0)      cfg 2 -> surf9 -2         cfg 3 -> surf9 -4, surf11 -2
# cfg 3 is the extreme design order.  Note the AXIS TRANSPOSE vs our chain
# nomenclature: our (m, n) = (-4, -2) puts -4 in x, Zemax cfg 3 puts -4 in y.
# The whole post-DOE system (surfaces 13-29) is rotationally symmetric about z
# -- every surface is a centred standard sphere -- so the two differ by a 90 deg
# rotation and NOT by any physics.  --swap-order writes the parameters directly
# instead of using the configs, so the transpose can be tested rather than
# assumed.
#
# POP centres its array on the CHIEF RAY of the selected field, which for a
# field-1 point object IS the diffracted order's chief ray.  That is what makes
# an off-axis order measurable at all: the array follows the order to
# (x, y) = (-960, -1920) um instead of staying on the mechanical axis.
#
# usage:
#   python pop_run_121.py --config 3 --nx 1024 --width 0.04 --tag base
import argparse
import json
import os
import sys
import time

import numpy as np

ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")
OUTDIR = (r"C:\Users\Tesla\AppData\Local\Temp\claude\C--Users-Tesla"
          r"\372a2d1f-acbe-4b57-a148-eeae3fe1d729\scratchpad\pop121")


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=int, default=3)
    p.add_argument('--nx', type=int, default=1024,
                   help='POP X and Y sampling (32..8192)')
    p.add_argument('--width', type=float, default=0.04,
                   help='initial array width at the start surface, mm')
    p.add_argument('--waist', type=float, default=0.004,
                   help='Gaussian waist radius at the start surface, mm')
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--end', type=int, default=29)
    p.add_argument('--field', type=int, default=1)
    p.add_argument('--auto-sampling', action='store_true',
                   help="call AutoCalculateBeamSampling (overrides --width)")
    p.add_argument('--swap-order', action='store_true',
                   help='write surf9 order = -2, surf11 order = -4 (the '
                        'transpose of cfg 3) to test the rotation assumption')
    p.add_argument('--m', type=int, default=None,
                   help='DOE order in GLOBAL X (written to surf 11 par2); '
                        'with --n this bypasses the config table entirely and '
                        'puts the order on the SAME axes our chain uses')
    p.add_argument('--n', type=int, default=None,
                   help='DOE order in GLOBAL Y (written to surf 9 par2)')
    p.add_argument('--crop-um', type=float, default=0.0,
                   help='save only +/- this many um of the image grid (the '
                        'full-array power is still recorded, so the '
                        'array-normalised EE stays exact)')
    p.add_argument('--last-resample', type=float, default=0.0,
                   help='if > 0, force a resample of the array at surface '
                        '--last-surf to this width (mm) before the final leg')
    p.add_argument('--last-surf', type=int, default=28)
    p.add_argument('--last-nx', type=int, default=0,
                   help='sampling to use at --last-surf (0 = keep --nx)')
    p.add_argument('--defocus', type=float, default=0.0,
                   help='shift the image surface by this many um along z '
                        '(added to surface 28 thickness) for through-focus')
    p.add_argument('--tag', default='run')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(OUTDIR, exist_ok=True)

    import zospy as zp
    from zospy.api import constants

    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")
    oss.load(ZMX)
    oss.MCE.SetCurrentConfiguration(args.config)

    par2 = constants.Editors.LDE.SurfaceColumn.Par2
    s9 = oss.LDE.GetSurfaceAt(9)
    s11 = oss.LDE.GetSurfaceAt(11)
    if args.swap_order:
        m9 = s9.GetSurfaceCell(par2).DoubleValue
        m11 = s11.GetSurfaceCell(par2).DoubleValue
        s9.GetSurfaceCell(par2).DoubleValue = m11
        s11.GetSurfaceCell(par2).DoubleValue = m9
    if args.m is not None or args.n is not None:
        # surf 11 deflects in GLOBAL X, surf 9 in GLOBAL Y (surfaces 10/12 are
        # +/-90 deg z-rotations around surf 11).  Verified by chief-ray trace.
        s11.GetSurfaceCell(par2).DoubleValue = float(args.m or 0)
        s9.GetSurfaceCell(par2).DoubleValue = float(args.n or 0)
    order = (s9.GetSurfaceCell(par2).DoubleValue,
             s11.GetSurfaceCell(par2).DoubleValue)

    s28 = oss.LDE.GetSurfaceAt(28)
    t28_0 = s28.Thickness
    if args.defocus:
        s28.Thickness = t28_0 + args.defocus * 1e-3

    # per-surface POP resample, only if asked for
    if args.last_resample > 0:
        pod = oss.LDE.GetSurfaceAt(args.last_surf).PhysicalOpticsData
        pod.AutoResample = False
        pod.XWidth = args.last_resample
        pod.YWidth = args.last_resample
        n = args.last_nx or args.nx
        pod.XSampling = constants.process_constant(
            constants.Editors.LDE.XYSampling, f"S_{n}x{n}")
        pod.YSampling = constants.process_constant(
            constants.Editors.LDE.XYSampling, f"S_{n}x{n}")

    # chief-ray intercept at the image plane (this is where POP centres)
    rt = oss.Tools.OpenBatchRayTrace()
    nr = rt.CreateNormUnpol(1, constants.Tools.RayTrace.RaysType.Real,
                            oss.LDE.NumberOfSurfaces - 1)
    nr.AddRay(1, 0.0, 0.0, 0.0, 0.0, constants.Tools.RayTrace.OPDMode.None_)
    rt.RunAndWaitForCompletion()
    nr.StartReadingResults()
    r = nr.ReadNextResult()
    chief = (float(r[4]), float(r[5]))
    rt.Close()

    pop = zp.analyses.physicaloptics.PhysicalOpticsPropagation(
        wavelength=1,
        field=args.field,
        start_surface=args.start,
        end_surface=args.end,
        surface_to_beam=0.0,
        use_polarization=False,
        separate_xy=False,
        beam_type='GaussianWaist',
        beam_parameters={'Waist X': args.waist, 'Waist Y': args.waist},
        x_sampling=f"{args.nx}x{args.nx}",
        y_sampling=f"{args.nx}x{args.nx}",
        x_width=args.width,
        y_width=args.width,
        use_total_power=True,
        total_power=1.0,
        use_peak_irradiance=False,
        data_type='Irradiance',
        project='AlongBeam',
        auto_calculate_beam_sampling=bool(args.auto_sampling),
    )
    t0 = time.time()
    res = pop.run(oss)
    dt = time.time() - t0

    df = res.data
    if df is None:
        print("POP returned NO data grid")
        for m in (res.messages or []):
            print("  msg:", m)
        oss.close()
        return 2

    I = np.asarray(df.values, dtype=float)          # [y, x], irradiance
    xs = np.asarray(df.columns, dtype=float)        # mm
    ys = np.asarray(df.index, dtype=float)          # mm
    dx = float(np.diff(xs).mean())
    dy = float(np.diff(ys).mean())
    P = float(I.sum()) * dx * dy

    head = list(res.header or [])
    msgs = [f"{m}" for m in (res.messages or [])]

    Isave, xsave, ysave = I, xs, ys
    if args.crop_um > 0:
        kx = np.abs(xs) <= args.crop_um * 1e-3
        ky = np.abs(ys) <= args.crop_um * 1e-3
        Isave = I[np.ix_(ky, kx)]
        xsave, ysave = xs[kx], ys[ky]

    ordname = (f"m{args.m:+d}n{args.n:+d}".replace('+', 'p').replace('-', 'm')
               if args.m is not None or args.n is not None
               else f"cfg{args.config}")
    tag = (f"{args.tag}_{ordname}_n{args.nx}_w{args.width:g}"
           f"{'_swap' if args.swap_order else ''}"
           f"{'' if not args.defocus else f'_dz{args.defocus:+g}'}")
    path = os.path.join(OUTDIR, tag + ".npz")
    np.savez_compressed(path, I=Isave, x=xsave, y=ysave,
                        chief=np.array(chief), order=np.array(order),
                        meta=json.dumps({
                            'config': args.config, 'nx': args.nx,
                            'width': args.width, 'waist': args.waist,
                            'start': args.start, 'end': args.end,
                            'field': args.field, 'swap': args.swap_order,
                            'm': args.m, 'n': args.n,
                            'defocus_um': args.defocus,
                            'crop_um': args.crop_um,
                            'P_full': P, 'dx_mm': dx, 'dy_mm': dy,
                            'full_shape': list(I.shape),
                            'full_extent_um': [float(xs[0] * 1e3),
                                               float(xs[-1] * 1e3)],
                            'last_resample': args.last_resample,
                            'auto_sampling': bool(args.auto_sampling),
                            'runtime_s': dt, 'header': head, 'messages': msgs,
                        }))

    print(f"=== POP cfg{args.config} order=({order[0]:+.0f} on s9 / "
          f"{order[1]:+.0f} on s11)  N={args.nx}  W={args.width} mm "
          f"waist={args.waist} mm  dz={args.defocus:+g} um ===")
    print(f"  runtime {dt:.1f} s")
    print(f"  chief ray at image: ({chief[0] * 1e3:+.3f}, "
          f"{chief[1] * 1e3:+.3f}) um")
    print(f"  grid {I.shape[0]}x{I.shape[1]}   dx={dx * 1e3:.5f} um  "
          f"dy={dy * 1e3:.5f} um   extent x[{xs[0] * 1e3:+.2f},"
          f"{xs[-1] * 1e3:+.2f}] y[{ys[0] * 1e3:+.2f},{ys[-1] * 1e3:+.2f}] um")
    print(f"  SUM(I)*dA = {P:.6f}  (source total power set to 1.0)")
    print(f"  peak I = {I.max():.6e}")
    for line in head:
        print("  [hdr]", line)
    for m in msgs:
        print("  [msg]", m)
    print("  saved ->", path)

    if args.defocus:
        s28.Thickness = t28_0
    oss.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())

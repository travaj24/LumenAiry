# POP CROSSCHECK, step 5: the TIE-BREAKER.
#
# POP says design 121's extreme order loses ~7 EE3 points against the on-axis
# order.  Our exact-ray + Rayleigh-Sommerfeld oracle says it loses ~0.  Both
# cannot be right, and POP is exactly the tool one distrusts off axis at high
# NA -- so the argument has to be settled by something that is NEITHER.
#
# Zemax's RMS WAVEFRONT ERROR is that something.  It is a pure ray computation
# (real ray OPD across the pupil against a reference sphere), it has no grid,
# no propagator and no sampling knob that can fake an answer, and it is the
# standard measure of whether a design is diffraction limited.  If cfg 3's RMS
# OPD is a small fraction of a wave, the DESIGN is diffraction limited at the
# extreme order, the oracle's "no degradation" is right and POP's deficit is a
# POP artefact.  If it is a large fraction of a wave, the oracle is missing
# real aberration.
#
# The Huygens PSF Strehl is reported alongside as a second, independent,
# non-POP witness (Huygens integrates the real ray OPD directly to the image
# plane; it shares the ray trace with the wavefront map but not the diffraction
# machinery).
#
# usage:  python pop_wfe_121.py [--orders 0,0 -1,0 -2,0 -3,0 -4,0 -4,-2]
import argparse
import sys

import numpy as np

ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--orders', nargs='+',
                   default=['0,0', '-1,0', '-2,0', '-3,0', '-4,0', '-4,-2'])
    p.add_argument('--sampling', default='128x128')
    a = p.parse_args(argv)

    import zospy as zp
    from zospy.api import constants

    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")
    oss.load(ZMX)
    par2 = constants.Editors.LDE.SurfaceColumn.Par2
    s9 = oss.LDE.GetSurfaceAt(9)
    s11 = oss.LDE.GetSurfaceAt(11)

    print("order      tilt [mrad]        RMS OPD [waves]   PV OPD [waves]   "
          "pupil pts")
    for spec in a.orders:
        m, n = (int(v) for v in spec.split(','))
        s11.GetSurfaceCell(par2).DoubleValue = float(m)   # global X
        s9.GetSurfaceCell(par2).DoubleValue = float(n)    # global Y
        wm = zp.analyses.wavefront.WavefrontMap(
            field=1, wavelength=1, surface='Image',
            sampling=a.sampling, use_exit_pupil=False,
            remove_tilt=False, sub_aperture_x=0.0, sub_aperture_y=0.0,
            sub_aperture_r=1.0)
        res = wm.run(oss)
        W = np.asarray(res.data.values, dtype=float)
        ok = np.isfinite(W)
        w = W[ok]
        # the map is in waves, referenced to the chief ray; piston and tilt are
        # not aberrations here (the chief ray defines the reference point), so
        # remove piston only -- Zemax's own RMS-to-centroid removes tilt too,
        # both are printed.
        rms_pist = float(np.sqrt(np.mean((w - w.mean()) ** 2)))
        pv = float(w.max() - w.min())
        # remove best-fit tilt as well
        ny, nx = W.shape
        gy, gx = np.mgrid[0:ny, 0:nx]
        A = np.column_stack([np.ones(w.size), gx[ok].ravel(), gy[ok].ravel()])
        c, *_ = np.linalg.lstsq(A, w, rcond=None)
        rms_tilt = float(np.sqrt(np.mean((w - A @ c) ** 2)))
        print(f"({m:+d},{n:+d})  ({m * 11.5158:+8.3f},{n * 11.5158:+8.3f})   "
              f"{rms_pist:10.5f} (piston-free)  {rms_tilt:9.5f} (tilt-free)  "
              f"{pv:9.4f}   {int(ok.sum())}")

    print("\nHuygens PSF Strehl (independent of POP, shares only the ray "
          "trace):")
    for spec in a.orders:
        m, n = (int(v) for v in spec.split(','))
        s11.GetSurfaceCell(par2).DoubleValue = float(m)
        s9.GetSurfaceCell(par2).DoubleValue = float(n)
        try:
            hp = zp.analyses.psf.HuygensPsf(
                pupil_sampling='64x64', image_sampling='64x64',
                image_delta=0.05, field=1, wavelength=1, normalize=False)
            r = hp.run(oss)
            I = np.asarray(r.data.values, dtype=float)
            print(f"  ({m:+d},{n:+d})  peak = {I.max():.6e}   "
                  f"(relative to the (0,0) peak this is the Strehl-like ratio)")
        except Exception as exc:                              # noqa: BLE001
            print(f"  ({m:+d},{n:+d})  Huygens PSF failed: {exc}")
    oss.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())

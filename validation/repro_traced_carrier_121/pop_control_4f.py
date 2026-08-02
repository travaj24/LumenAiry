# POP CROSSCHECK, control: reproduce the ONE recorded Zemax number in the tree.
#
# The brief asks the (0,0) POP to "reproduce the recorded 2.737 um waist radius
# / 3.223 um FWHM".  Reading where that number came from
# (docs/audit_asm_thinlens_focus_2026_07_18.md, table row
# "colleague's GUI POP | paraxial x pilot-beam POP | 2.7378 | clean to 1e-4"),
# it is NOT a POP of the 121 prescription at all.  It is a POP of the
# PARAXIALLY EQUIVALENT 4f system, f1 = 60.916 mm, f2 = 41.666 mm, launched
# with the same 4 um waist: a pair of IDEAL paraxial lenses with no aberration
# and no real glass, whose answer is just the magnification,
# 4 um x f2/f1 = 2.736 um.
#
# So that number cannot gate a POP of the real design -- the real design has
# aberration and a real exit NA, and this campaign's OWN ideal-field ceiling
# for the same readout is 3.45-3.55 um FWHM, not 3.223.  What the number CAN
# do, and what this script uses it for, is validate the POP DRIVING: if my
# ZOS-API POP setup reproduces 2.7378 um on the 4f, the setup is right and any
# difference on the real system is physics rather than harness error.
#
# usage:  python pop_control_4f.py [--nx 2048] [--width 0.2]
import argparse
import sys

import numpy as np

F1 = 60.916
F2 = 41.666
LAM_UM = 1.31
W0_MM = 0.004


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--nx', type=int, default=2048)
    p.add_argument('--width', type=float, nargs='+', default=[0.1, 0.2, 0.4])
    p.add_argument('--f1', type=float, default=F1)
    p.add_argument('--f2', type=float, default=F2)
    a = p.parse_args(argv)

    import zospy as zp
    from zospy.api import constants

    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")
    oss.new()
    sysd = oss.SystemData
    sysd.Wavelengths.GetWavelength(1).Wavelength = LAM_UM
    sysd.Aperture.ApertureType = constants.SystemData.ZemaxApertureType.ObjectSpaceNA
    sysd.Aperture.ApertureValue = 0.21

    # 0 OBJ | 1 waist/dummy | 2 -> f1 | 3 paraxial f1 | 4 -> f1+f2 |
    # 5 paraxial f2 | 6 -> f2 | 7 IMA
    lde = oss.LDE
    while lde.NumberOfSurfaces < 8:
        lde.InsertNewSurfaceAt(1)
    lde.GetSurfaceAt(0).Thickness = 0.0
    lde.GetSurfaceAt(1).Thickness = a.f1
    s2 = lde.GetSurfaceAt(2)
    s2.ChangeType(s2.GetSurfaceTypeSettings(
        constants.Editors.LDE.SurfaceType.Paraxial))
    s2.GetSurfaceCell(constants.Editors.LDE.SurfaceColumn.Par1
                      ).DoubleValue = a.f1          # focal length
    s2.GetSurfaceCell(constants.Editors.LDE.SurfaceColumn.Par2
                      ).IntegerValue = 1            # OPD mode (integer cell)
    s2.Thickness = a.f1 + a.f2
    s3 = lde.GetSurfaceAt(3)
    s3.ChangeType(s3.GetSurfaceTypeSettings(
        constants.Editors.LDE.SurfaceType.Paraxial))
    s3.GetSurfaceCell(constants.Editors.LDE.SurfaceColumn.Par1
                      ).DoubleValue = a.f2
    s3.GetSurfaceCell(constants.Editors.LDE.SurfaceColumn.Par2
                      ).IntegerValue = 1
    s3.Thickness = a.f2
    for s in range(1, lde.NumberOfSurfaces):
        lde.GetSurfaceAt(s).SemiDiameter = 30.0

    print(f"paraxial 4f: f1={a.f1} f2={a.f2}  m = f2/f1 = {a.f2 / a.f1:.6f}")
    print(f"analytic image waist = {W0_MM * 1e3 * a.f2 / a.f1:.4f} um   "
          f"-> FWHM {W0_MM * 1e3 * a.f2 / a.f1 * np.sqrt(2 * np.log(2)):.4f} um"
          f"   (recorded colleague POP: 2.7378 um / 3.223 um)")
    print("\n   N     W[mm]   dx_img[um]   POP BeamWidth[um]   measured w[um]"
          "   FWHM[um]")
    for W in a.width:
        pop = zp.analyses.physicaloptics.PhysicalOpticsPropagation(
            wavelength=1, field=1, start_surface=1, end_surface='Image',
            beam_type='GaussianWaist',
            beam_parameters={'Waist X': W0_MM, 'Waist Y': W0_MM},
            x_sampling=f"{a.nx}x{a.nx}", y_sampling=f"{a.nx}x{a.nx}",
            x_width=W, y_width=W,
            use_total_power=True, total_power=1.0, use_peak_irradiance=False,
            data_type='Irradiance', project='AlongBeam')
        res = pop.run(oss)
        df = res.data
        I = np.asarray(df.values, float)
        xs = np.asarray(df.columns, float) * 1e3      # um
        ys = np.asarray(df.index, float) * 1e3
        dx = float(np.diff(xs).mean())
        bw = [ln for ln in (res.header or []) if 'Beam Width' in ln]
        bwv = (float(bw[0].split('=')[1].split(',')[0]) * 1e3) if bw else np.nan
        # 1/e^2 radius by second moment, and FWHM by ring average
        X, Y = np.meshgrid(xs, ys)
        tot = I.sum()
        cx = (I * X).sum() / tot
        cy = (I * Y).sum() / tot
        r2 = ((X - cx) ** 2 + (Y - cy) ** 2)
        w_mom = np.sqrt(2 * (I * r2).sum() / tot)     # w = 2*sigma for Gaussian
        r = np.sqrt(r2)
        nb = int(min(I.shape[0] // 2, 12.0 / dx))
        ring = np.clip((r / dx).astype(int), 0, nb)
        s = np.bincount(ring.ravel(), weights=I.ravel(), minlength=nb + 1)
        cn = np.bincount(ring.ravel(), minlength=nb + 1)
        prof = s[:nb] / np.maximum(cn[:nb], 1)
        prof = prof / prof[0]
        idx = np.where(prof < 0.5)[0]
        i1 = idx[0]
        f = (prof[i1 - 1] - 0.5) / (prof[i1 - 1] - prof[i1])
        fwhm = 2 * ((i1 - 1 + 0.5) * dx + f * dx)
        # 1/e^2 radius straight off the ring profile
        j = np.where(prof < np.exp(-2))[0]
        we = np.nan
        if len(j) and j[0] > 0:
            i2 = j[0]
            g = (prof[i2 - 1] - np.exp(-2)) / (prof[i2 - 1] - prof[i2])
            we = (i2 - 1 + 0.5) * dx + g * dx
        print(f"  {a.nx:5d} {W:7.3f} {dx:11.5f} {bwv:17.4f} "
              f"{we:14.4f}   {fwhm:8.4f}   (2nd-moment w {w_mom:.4f})")
    oss.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())

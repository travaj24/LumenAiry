# POP CROSSCHECK, step 0: can we talk to Zemax at all, and does the .zmx
# expose the DOE order the way we think it does?
#
# This script makes NO physics claim.  It only:
#   1. opens a STANDALONE ZOS-API connection (zospy 2.1.5, the idiom proven in
#      OPDPy_Lumenairy_Crosscheck/zos_gauss_121.py),
#   2. loads the exact design file the chain loads,
#   3. dumps the multi-configuration table so we can see WHICH config selects
#      which (m, n) DOE order on surfaces 9 and 11,
#   4. traces the field-1 chief ray to the image plane in every config -- that
#      intercept is where POP will centre its array, and it must equal the
#      exact-ray oracle's chief-ray intercept for the comparison to be
#      apples-to-apples,
#   5. reports where OpticStudio keeps its POP beam files.
#
# Run with the zemax venv:
#   D:\...\OPDPy_Lumenairy_Crosscheck\.venv-zemax\Scripts\python.exe pop_probe_setup.py
import sys
import traceback

ZMX = (r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics"
       r"\Reverse_Symmetric_ASM\tx4designstudy121\20260707 dll Tx02-MSOP16.zmx")


def main():
    import zospy as zp

    print("zospy", zp.__version__)
    zos = zp.ZOS()
    oss = zos.connect(mode="standalone")
    print("OpticStudio :", zos.version)
    print("licence     :", zos.Application.LicenseStatus)
    print("objects dir :", zos.Application.ObjectsDir)

    oss.load(ZMX)
    print("file        :", oss.SystemFile)
    print("mode        :", oss.Mode)
    print("surfaces    :", oss.LDE.NumberOfSurfaces - 1, "(0..N)")
    sysd = oss.SystemData
    print("wavelengths :", sysd.Wavelengths.NumberOfWavelengths,
          "  w1 =", sysd.Wavelengths.GetWavelength(1).Wavelength, "um")
    print("aperture    :", sysd.Aperture.ApertureType,
          sysd.Aperture.ApertureValue)
    nf = sysd.Fields.NumberOfFields
    print("fields      :", nf, sysd.Fields.GetFieldType())
    for i in range(1, nf + 1):
        f = sysd.Fields.GetField(i)
        print(f"   field {i}: X={f.X}  Y={f.Y}  weight={f.Weight}")

    mce = oss.MCE
    ncfg = mce.NumberOfConfigurations
    print(f"\nconfigs     : {ncfg}   current = {mce.CurrentConfiguration}")

    import zospy.api.config  # noqa: F401  (ensures constants are loaded)
    from zospy.api import constants

    print("\n--- surface types ---")
    for s in range(oss.LDE.NumberOfSurfaces):
        surf = oss.LDE.GetSurfaceAt(s)
        try:
            tname = surf.TypeName
        except Exception:
            tname = "?"
        print(f"  {s:3d} {tname:<12} thick={surf.Thickness:14.8f} "
              f"semi={surf.SemiDiameter:9.5f}  {surf.Comment}")

    print("\n--- DOE order per configuration (surf 9 = x, surf 11 = y) ---")
    rows = {}
    for c in range(1, ncfg + 1):
        mce.SetCurrentConfiguration(c)
        s9 = oss.LDE.GetSurfaceAt(9)
        s11 = oss.LDE.GetSurfaceAt(11)
        # DGRATING: par1 = lines/um (signed), par2 = diffraction order
        f9 = s9.GetSurfaceCell(constants.Editors.LDE.SurfaceColumn.Par1).DoubleValue
        m9 = s9.GetSurfaceCell(constants.Editors.LDE.SurfaceColumn.Par2).DoubleValue
        f11 = s11.GetSurfaceCell(constants.Editors.LDE.SurfaceColumn.Par1).DoubleValue
        m11 = s11.GetSurfaceCell(constants.Editors.LDE.SurfaceColumn.Par2).DoubleValue
        rows[c] = (f9, m9, f11, m11)
        print(f"  cfg {c}: surf9  freq={f9:+.6f} l/um  order={m9:+.1f}   "
              f"surf11 freq={f11:+.6f} l/um  order={m11:+.1f}")

    # chief ray (Px=Py=0) of field 1 to the image surface, per config
    print("\n--- field-1 chief ray at the image plane, per config ---")
    lam_um = sysd.Wavelengths.GetWavelength(1).Wavelength
    for c in range(1, ncfg + 1):
        mce.SetCurrentConfiguration(c)
        try:
            rt = oss.Tools.OpenBatchRayTrace()
            nsurf = oss.LDE.NumberOfSurfaces - 1
            nr = rt.CreateNormUnpol(1, constants.Tools.RayTrace.RaysType.Real,
                                    nsurf)
            nr.AddRay(1, 0.0, 0.0, 0.0, 0.0,
                      constants.Tools.RayTrace.OPDMode.None_)
            rt.RunAndWaitForCompletion()
            nr.StartReadingResults()
            res = nr.ReadNextResult()
            ok = res[0]
            _, _, err, vig, x, y, z, L, M, N, l2, m2, n2, opd, I = res
            rt.Close()
            f9, m9, f11, m11 = rows[c]
            # analytic prediction: sin(theta) = m * lambda * nu
            tx = m9 * lam_um * f9
            ty = m11 * lam_um * f11
            print(f"  cfg {c}: ok={ok} err={err} vig={vig}  "
                  f"image (x,y) = ({x * 1e3:+9.3f}, {y * 1e3:+9.3f}) um   "
                  f"dir (L,M) = ({L:+.5f},{M:+.5f})  I={I:.4f}")
            print(f"          DOE tilt from (m*lam*nu): "
                  f"({tx * 1e3:+.3f}, {ty * 1e3:+.3f}) mrad "
                  f"[surf-9 local frame; surf 11 is rotated +90 deg about z]")
        except Exception:
            traceback.print_exc()

    mce.SetCurrentConfiguration(1)
    print("\nPOP beam-file folder should be "
          f"{zos.Application.ObjectsDir}\\POP\\BEAMFILES")
    oss.close()
    print("DONE")


if __name__ == "__main__":
    sys.exit(main())

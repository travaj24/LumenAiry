"""Probe 12: numexpr-present vs numexpr-absent for a complex64 field.
numexpr is NOT installed here, so emulate the two code paths exactly."""
import numpy as np
lam = 632.8e-9; k0 = 2*np.pi/lam
rng = np.random.default_rng(1)
N = 1024                       # E.size = 1 Mi == _NUMEXPR_MIN_SIZE, so the gate fires
E64 = (rng.random((N, N)) + 1j*rng.random((N, N))).astype(np.complex64)
for depth_nm, tag in ((2e-4, 'sag 200 um'), (5e-3, 'piston-scale 5 mm')):
    opd = rng.random((N, N)) * depth_nm
    # numpy fallback (what runs WITHOUT numexpr): exp in complex128, cast, multiply
    ph = np.exp(-1j*k0*opd)
    A = E64 * ph.astype(np.complex64)
    # numexpr path (out=E): complex128 internally, cast only at the store
    B = (E64.astype(np.complex128) * ph).astype(np.complex64)
    d = np.abs(A.astype(np.complex128) - B.astype(np.complex128))
    m = np.abs(B).max()
    print(f"  {tag:<20} max|numpy - numexpr| / peak = {d.max()/m:.4e}   "
          f"rms = {np.sqrt(np.mean(d**2))/m:.4e}")
print("  => the two code paths are NOT bit-identical for complex64; which one")
print("     runs depends on whether numexpr is installed (and on E.size).")

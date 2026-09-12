"""Probe 7: odd-N centring consistency; complex64 preservation; dead code."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.propagators.carrier as C
from lumenairy.propagators.carrier import (
    propagate_carrier_referenced, carrier_referenced_reconstruct,
    carrier_referenced_envelope, _freq_sq_1d, _freq_sq_1d_bld, _asm_axis)

wl = 1.31e-6
print("=== dead code: _freq_sq_1d vs _freq_sq_1d_bld at odd N ===")
for N in (4, 5, 7):
    a = _freq_sq_1d(N, 1.0); b = _freq_sq_1d_bld(N, 1.0, np)
    print(f"  N={N}: _freq_sq_1d={np.round(a,4)}  _bld={np.round(b,4)}  same={np.allclose(a,b)}")

print("\n=== odd-N astigmatic path: bandlimit mask centring in _asm_axis ===")
for N in (64, 65):
    d = 2e-6
    E = np.zeros((N, N), complex); E[N//2, N//2] = 1.0
    out = _asm_axis(E, 1e-3, wl, d, axis=1, bandlimit=True)
    # the bandlimit mask uses (arange(N)-N/2) while H uses N//2 -> mismatch for odd N
    f_h  = (np.arange(N) - (N//2))/(N*d)
    f_bl = (np.arange(N) - N/2)/(N*d)
    print(f"  N={N}: H freq axis offset N//2={N//2}, bandlimit mask offset N/2={N/2}; "
          f"max|diff|={np.abs(f_h-f_bl).max():.4e} 1/m  ({'MISMATCH' if N%2 else 'ok'})")

print("\n=== odd-N scalar path: reconstruct/envelope round trip and grid centring ===")
for N in (64, 65):
    dx = 2e-6
    x = (np.arange(N)-N/2)*dx          # what _radial_carrier_phase uses
    x2 = (np.arange(N)-N//2)*dx        # what _freq_*_bld / fftfreq imply
    env = np.ones((N, N), complex)
    E = carrier_referenced_reconstruct(env, 1e-3, wl, dx)
    back = carrier_referenced_envelope(E, 1e-3, wl, dx)
    print(f"  N={N}: spatial-grid offset used by carrier phase = N/2 = {N/2}; "
          f"FFT-implied centre = N//2 = {N//2}; round-trip err={np.abs(back-env).max():.2e}")
    # is the carrier phase symmetric about a sample?  (a real diagnostic)
    ph = np.angle(E[N//2, :])
    print(f"        carrier phase row: min at index {int(np.argmin(np.abs(ph)))} (centre sample is {N//2})")

print("\n=== complex64 preservation ===")
N = 256; dx = 2e-6
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x, indexing='xy')
env64 = np.exp(-(X**2+Y**2)/(50e-6)**2).astype(np.complex64)
for R in (np.inf, 0.05):
    e, Ro, dxo = propagate_carrier_referenced(env64, R, 1e-3, wl, dx)
    print(f"  propagate_carrier_referenced R={R}: in c64 -> out {np.asarray(e).dtype}")
print(f"  reconstruct: {carrier_referenced_reconstruct(env64, 0.05, wl, dx).dtype}")
print(f"  envelope:    {carrier_referenced_envelope(env64, 0.05, wl, dx).dtype}")
from lumenairy.propagators.carrier_field import CarrierField, FieldGrid, CarrierSpec
cf = CarrierField(envelope=env64, grid=FieldGrid(shape=(N,N), dx=dx),
                  carrier=CarrierSpec(R=0.05), wavelength=wl)
print(f"  CarrierField.envelope dtype after __post_init__: {cf.envelope.dtype}")
print(f"  CarrierField.full_field() dtype: {cf.full_field().dtype}")
print(f"  CarrierSpec.phasor_on dtype: {cf.carrier.phasor_on(cf.grid, wl).dtype}")

import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements import _lens_traced as LT

# --- (a) partition of unity of _flattop_partition_1d -------------------
u = np.linspace(-1, 1, 20001)
for cuts, hw in [([0.0], 0.2), ([-0.4, 0.3], 0.1), ([], 0.2),
                 ([-0.5, -0.1, 0.4], 0.05), ([0.0], 0.6)]:
    W = LT._flattop_partition_1d(u, cuts, hw)
    S = sum(W)
    print('cuts=%-22s hw=%.2f  K=%d  max|sum-1| = %.3e' %
          (str(cuts), hw, len(W), float(np.abs(S-1).max())))

# --- (b) degenerate inputs to the helpers ------------------------------
fr = np.fft.fftshift(np.fft.fftfreq(64, 1e-5))
print('occ(all zero)  ->', LT._occupied_freq_support(np.zeros(64), fr, 0.995))
print('occ(single)    ->', LT._occupied_freq_support(
    np.eye(64)[32], fr, 0.995))
print('cuts(all zero) ->', LT._spectral_gap_cuts(np.zeros(64), fr, fr[0], fr[-1], .15, .25))
print('cuts(single pk)->', LT._spectral_gap_cuts(np.eye(64)[32], fr, fr[0], fr[-1], .15, .25))
try:
    print('occ(empty)     ->', LT._occupied_freq_support(np.zeros(0), np.zeros(0), .995))
except Exception as e:
    print('occ(empty)     -> %s: %s' % (type(e).__name__, e))
try:
    print('cuts(empty)    ->', LT._spectral_gap_cuts(np.zeros(0), np.zeros(0), 0, 1, .15, .25))
except Exception as e:
    print('cuts(empty)    -> %s: %s' % (type(e).__name__, e))

# --- (c) segments sum to E exactly with min_segment_power=0 ------------
N, dx = 256, 6e-6
def two_beam(N, dx, th):
    x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
    k = 2*np.pi/fixt.WL
    g = np.exp(-(X**2+Y**2)/(0.25e-3)**2)
    return (g*np.exp(1j*k*th*X) + g*np.exp(-1j*k*th*X)).astype(np.complex128)
for label, E in [('two-beam +/-25 mrad', two_beam(N, dx, 25e-3)),
                 ('single gaussian', fixt.gauss(N, dx, 0.25e-3)),
                 ('all zero', np.zeros((N,N), complex))]:
    for msp in (0.0, 1e-3):
        segs = LT.apply_real_lens_traced_segmented(
            E, prescription=fixt.small_singlet(), wavelength=fixt.WL, dx=dx,
            min_segment_power=msp, return_segments=True)
        S = sum(segs)
        num = float(np.abs(S-E).max()); den = float(np.abs(E).max()) or 1.0
        print('%-22s msp=%-6g nseg=%2d  max|sum-E|/max|E| = %.3e' %
              (label, msp, len(segs), num/den))

# ADVERSARIAL REVIEW - attack line A.  The audit's mechanism claim: "an aliased
# sphere cannot diffract/diverge on the grid, so the wave pass transports the
# amplitude through the group essentially UNCHANGED".  Test it in isolation:
# a Gaussian x EXACT sphere(R) propagated a short distance by the same band-
# limited ASM apply_real_lens uses for its in-glass legs, at grid pitches that
# put the carrier alias radius well inside / near / outside the beam.
# Ground truth (geometric, |R| >> zR): w_out/w_in = |1 + d/R|.
import os, sys, warnings
import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, r"D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy")
import lumenairy as la

lam = 1.31e-6
k0 = 2 * np.pi / lam
w = 5.0e-3


def run(R, d, N, win, label):
    dx = win / N
    x = (np.arange(N) - N // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    env = np.exp(-r2 / w ** 2)
    S = np.sign(R) * (np.sqrt(r2 + R * R) - abs(R))
    E = (env * np.exp(1j * k0 * S)).astype(np.complex128)
    r_al = abs(R) * lam / (2 * dx)
    E2 = la.angular_spectrum_propagate(E, d, lam, dx, bandlimit=True)
    E2 = np.asarray(E2)

    def w2m(F, rmax):
        I = np.abs(F) ** 2
        msk = r2 <= rmax ** 2
        tot = I[msk].sum()
        return np.sqrt(2.0 * (I * r2)[msk].sum() / tot), tot

    w_i, P_i = w2m(E, 2.0 * w)
    w_o, P_o = w2m(E2, 2.0 * w)
    Ptot_i = (np.abs(E) ** 2).sum()
    Ptot_o = (np.abs(E2) ** 2).sum()
    # radial profile check: is the exit still a clean Gaussian, or junk?
    I = np.abs(E2) ** 2
    c = N // 2
    prof = I[c, c:]
    rr = x[c:]
    sel = (rr < 1.6 * w) & (prof > 0)
    # fit ln I = a - 2 r^2 / w_fit^2
    p = np.polyfit(rr[sel] ** 2, np.log(prof[sel]), 1)
    w_fit = np.sqrt(-2.0 / p[0])
    print(f"{label:<22}{N:>6}{dx*1e6:>9.2f}{r_al*1e3:>9.3f}{r_al/w:>8.2f}"
          f"{w_o/w_i:>11.4f}{w_fit/w:>11.4f}{P_o/P_i:>10.4f}"
          f"{Ptot_o/Ptot_i:>10.6f}")


print(f"{'case':<22}{'N':>6}{'dx um':>9}{'r_alias':>9}{'/w':>8}"
      f"{'w2m ratio':>11}{'wfit/w_in':>11}{'P(r<2w)':>10}{'P_tot':>10}")
for R, d, tag in ((+47.906e-3, 4.40e-3, 'DIVERGE R=+47.9'),
                  (-21.139e-3, 4.40e-3, 'CONVERGE R=-21.1')):
    truth = abs(1 + d / R)
    print(f"  --- {tag}, d={d*1e3:.2f}mm, geometric truth w_out/w_in = "
          f"{truth:.4f} ---")
    for N, win in ((1024, 42.0e-3), (2048, 42.0e-3), (4096, 42.0e-3),
                   (8192, 42.0e-3), (8192, 21.0e-3), (8192, 10.5e-3)):
        if win / N * 1e6 < 0.6:
            continue
        run(R, d, N, win, f'  {tag[:8]} win{win*1e3:.1f}')

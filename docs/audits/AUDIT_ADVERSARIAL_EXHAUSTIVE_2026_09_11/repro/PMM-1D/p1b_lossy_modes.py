"""PROBE 1b: LOSSY (complex-root) lamellar modes, near-degenerate roots, and
the sqrt BRANCH rule.

Test A: every SEM eigenvalue must be a ROOT of the exact Botten dispersion
        relation  ->  residual |F(u_sem)| / scale.
Test B: every exact root found by an independent complex Newton sweep must have
        a matching SEM eigenvalue  ->  no MISSED mode.
Test C: the forward branch: after _forward_branch_flip, Im(q) >= 0 (or q real
        with Re(q) > 0) for every mode -- the exp(+i q k0 z) decay rule.
Test D: an ON-CUT case (q exactly real-negative / purely imaginary) and the
        claim "the on-cut flip is -r, not conj(r)".
"""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc

wl = 1.0e-6
k0 = 2 * np.pi / wl


def F_factory(eps1, eps2, f, period, kx0, pol):
    d1, d2 = f * period, (1 - f) * period

    def F(u):
        k1 = np.sqrt(complex(k0 * k0 * (eps1 - u)))
        k2 = np.sqrt(complex(k0 * k0 * (eps2 - u)))
        if abs(k1) < 1e-30:
            k1 = 1e-30
        if abs(k2) < 1e-30:
            k2 = 1e-30
        a = (k1 / k2) if pol == "te" else (k1 * eps2) / (k2 * eps1)
        return (np.cos(k1 * d1) * np.cos(k2 * d2)
                - 0.5 * (a + 1.0 / a) * np.sin(k1 * d1) * np.sin(k2 * d2)
                - np.cos(kx0 * period))
    return F


CASES = [
    ("Si/air   P=lam  f=0.5  normal", 3.48**2, 1.0, 0.5, 1.0, 0.0),
    ("Au/air   P=lam  f=0.5  normal", (0.18 + 3.43j)**2, 1.0, 0.5, 1.0, 0.0),
    ("Au/air   P=lam  f=0.5  25deg ", (0.18 + 3.43j)**2, 1.0, 0.5, 1.0, 25.0),
    ("Au/Si    P=2lam f=0.3  40deg ", (0.18 + 3.43j)**2, 3.48**2, 0.3, 2.0,
     40.0),
    ("Si/air   P=lam  f=0.9999 (near-degenerate limit)", 3.48**2, 1.0, 0.9999,
     1.0, 0.0),
]

for label, e1, e2, f, pl, angd in CASES:
    period = pl * wl
    kx0 = np.sin(np.deg2rad(angd)) * k0
    print("=" * 76)
    print(label)
    for pol in ("te", "tm"):
        mats = pc._build_sem(period, f * period, e1, e2, 28, 1, 1, False)
        W, lam, q, invop = pc._sem_modes(mats, k0, pol, kx0, False)
        u = q ** 2
        F = F_factory(e1, e2, f, period, kx0, pol)
        # Test A: residual of the exact relation at each SEM eigenvalue,
        # normalised by the local derivative scale so it reads as a distance
        res, dist = [], []
        for v in u:
            fv = F(v)
            h = 1e-7 * max(abs(v), 1.0)
            dfv = (F(v + h) - F(v - h)) / (2 * h)
            res.append(abs(fv))
            dist.append(abs(fv) / max(abs(dfv), 1e-300))   # ~ |u - u_root|
        res, dist = np.array(res), np.array(dist)
        # sort by |u| (low modes first)
        o = np.argsort(np.abs(u))
        k = min(12, len(u))
        print(f"  {pol}: {len(u)} SEM modes; lowest {k} by |u|:")
        print(f"     max |F(u_sem)|/|F'| over lowest {k} = "
              f"{dist[o[:k]].max():.3e}   (over ALL modes "
              f"{dist.max():.3e})")
        # Test B: Newton from a dense complex seed grid -> distinct roots, then
        # check each is matched by a SEM eigenvalue
        seeds = []
        emax = max(abs(e1), abs(e2))
        for rr in np.linspace(-40, float(np.real(emax)) + 2, 60):
            for ii in np.linspace(-25, 25, 25):
                seeds.append(complex(rr, ii))
        roots = []
        for s0 in seeds:
            z = s0
            ok = True
            for _ in range(60):
                h = 1e-8 * max(abs(z), 1.0)
                fz = F(z)
                dz = (F(z + h) - F(z - h)) / (2 * h)
                if dz == 0 or not np.isfinite(abs(dz)):
                    ok = False
                    break
                step = fz / dz
                z = z - step
                if abs(step) < 1e-13 * max(abs(z), 1.0):
                    break
            else:
                ok = abs(F(z)) < 1e-6
            if not ok or not np.isfinite(abs(z)):
                continue
            if abs(z.real) > 60 or abs(z.imag) > 40:
                continue
            if abs(F(z)) > 1e-7 * max(1.0, abs(z)):
                continue
            if not any(abs(z - r) < 1e-7 * max(1.0, abs(z)) for r in roots):
                roots.append(z)
        roots = sorted(roots, key=lambda z: -abs(z.real) * 0 - z.real)
        miss = []
        for r in roots[:14]:
            d = np.min(np.abs(u - r))
            if d > 1e-6 * max(1.0, abs(r)):
                miss.append((r, d))
        print(f"     independent Newton found {len(roots)} distinct roots; "
              f"of the 14 with largest Re(u): {len(miss)} unmatched by SEM")
        for r, d in miss[:5]:
            print(f"        MISSED root u={r:.8f}  nearest SEM at {d:.3e}")
        # Test C: forward branch
        bad = [(i, qq) for i, qq in enumerate(q)
               if not (qq.imag > -1e-8 * max(abs(qq), 1.0))]
        realish = [qq for qq in q if abs(qq.imag) <= 1e-8 * max(abs(qq), 1.0)]
        badre = [qq for qq in realish if qq.real < 0]
        print(f"     forward branch: {len(bad)} modes with Im(q) < 0, "
              f"{len(badre)} near-real modes with Re(q) < 0")
        if bad[:3]:
            print("        e.g.", [f"{qq:.4g}" for _i, qq in bad[:3]])

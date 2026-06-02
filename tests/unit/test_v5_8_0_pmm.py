"""v5.8.0 Polynomial Modal Method (PMM, item G) -- 1-D lamellar grating.

A non-Fourier modal solver (subsectional spectral-element / Edee) that converges
spectrally in the polynomial degree with NO accuracy floor -- the property the
ASR coordinate stretch (v5.7) lacked.  Validation anchors:

* the FMM solver (``rcwa_efficiency_1d``) at high order is the convergence
  oracle (PMM must reach the SAME efficiencies),
* a uniform ABSORBING slab must match the FMM slab bit-for-bit (the public<->
  engineering convention discipline -- the worst pitfall during development),
* lossless gratings conserve energy, and PMM keeps converging (no floor).
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d

# gold-like lossy metal ridge (PUBLIC Im > 0) -- the canonical hard TM case
GOLD = dict(period=1.0e-6, n_ridge=0.2 + 6j, n_groove=1.0, n_substrate=1.0,
            n_superstrate=1.0, depth=0.3e-6, duty_cycle=0.5, wavelength=0.8e-6)


def _oracle_T(pol, M=80):
    o, R, T = rcwa_efficiency_1d(**GOLD, polarization=pol, n_orders=M)
    return float(T.sum())


def _pmm_T(pol, degree, **kw):
    o, R, T = pmm_efficiency_1d(**GOLD, polarization=pol, degree=degree, **kw)
    return float(T.sum())


# ---------------------------------------------------------------------------
# converges to the FMM oracle
# ---------------------------------------------------------------------------

def test_pmm_te_converges_to_fmm_oracle():
    assert abs(_pmm_T("te", 16) - _oracle_T("te")) < 5e-5


def test_pmm_tm_converges_to_fmm_oracle():
    assert abs(_pmm_T("tm", 28) - _oracle_T("tm")) < 5e-4


# ---------------------------------------------------------------------------
# NO FLOOR -- the decisive property over ASR.  Error improves with degree, and
# PMM self-converges (consecutive high degrees agree), i.e. no plateau.
# ---------------------------------------------------------------------------

def test_pmm_tm_error_improves_with_degree():
    ref = _oracle_T("tm")
    e_low = abs(_pmm_T("tm", 8) - ref)
    e_high = abs(_pmm_T("tm", 28) - ref)
    assert e_high < e_low                       # monotone improvement (no floor)


def test_pmm_self_converges_no_floor():
    # independent of the oracle's own residual: two high degrees must agree,
    # proving PMM keeps converging rather than plateauing with noise.  (Robust
    # across LAPACK builds because the stabilize selector returns a clean,
    # resonance-free solve at each degree -- see the regression test below.)
    assert abs(_pmm_T("tm", 24) - _pmm_T("tm", 36)) < 1e-4
    assert abs(_pmm_T("te", 16) - _pmm_T("te", 28)) < 1e-5


def test_pmm_degree_robustness_no_resonance_leak():
    """Regression (CI flake): PMM has discrete resonances at isolated polynomial
    degrees where a near-singular interface mode-match injects spurious flux,
    inflating ``sum(R)+sum(T)`` and biasing the efficiencies.  Their location
    shifts with the LAPACK build, so a fixed degree was not CI-portable --
    ``degree=24`` returned an off-curve ``T`` (0.3997 vs the true 0.3982) on the
    CI build while passing locally.  The ``stabilize`` selector must therefore
    return a clean, resonance-free solve at EVERY requested degree: no inflated
    total power and no off-curve efficiency outlier.
    """
    for pol in ("tm", "te"):
        ref = _oracle_T(pol)                         # FMM truth anchor (no self-ref)
        for d in range(12, 41):
            o, R, T = pmm_efficiency_1d(**GOLD, polarization=pol, degree=d)
            rt = float(R.sum() + T.sum())
            assert rt <= 1.0 + 1e-3, (
                f"{pol} degree {d}: R+T={rt:.5f} inflated -- a resonance leaked "
                f"through the stabilize selector")
            # anchor to the FMM oracle (bites for BOTH pols, unlike a self-median:
            # a leaked resonant/under-converged degree lands >~1e-3 off truth,
            # while the genuine convergence drift over 12..40 stays ~1e-4).
            assert abs(float(T.sum()) - ref) < 1e-3, (
                f"{pol} degree {d}: T={float(T.sum()):.6f} is {abs(float(T.sum())-ref):.2e} "
                f"off the FMM oracle {ref:.6f} -- a resonant/under-converged "
                f"degree leaked through stabilize")


def test_pmm_low_degree_high_index_converges():
    """Regression (adversarial audit): a min-power 'clean = lowest total' rule
    silently returned the WORST-converged low-degree solve on high-index gratings
    (where the field is under-resolved and R+T sits *below* its converged value).
    The consensus selector must instead track the converged value even when a low
    degree is requested.  Silicon ridge n~=4 was biased by up to 6.5e-2 under the
    naive rule; it must now match the FMM oracle.
    """
    si = dict(period=1.2e-6, n_ridge=4.0 + 0.1j, n_groove=1.0, n_substrate=3.5,
              n_superstrate=1.0, depth=0.4e-6, duty_cycle=0.6, wavelength=1.0e-6)
    for pol in ("te", "tm"):
        o, R, T = rcwa_efficiency_1d(**si, polarization=pol, n_orders=160)
        ref = float(T.sum())
        for d in (8, 10, 12):
            op, Rp, Tp = pmm_efficiency_1d(**si, polarization=pol, degree=d)
            # the under-converged-deficit bug gave dT >~ 4e-3 (te up to 6.5e-2);
            # the consensus value tracks the oracle (te ~4e-5, tm ~1.4e-3 -- the
            # honest low-degree TM rate, still far from the broken deficit).
            assert abs(float(Tp.sum()) - ref) < 3e-3, (
                f"silicon {pol} degree {d}: T={float(Tp.sum()):.6f} is "
                f"{abs(float(Tp.sum())-ref):.2e} off the FMM oracle {ref:.6f} -- "
                f"the consensus selector returned an under-converged solve")


def test_pmm_beats_fmm_at_matched_dof_te():
    # PMM-TE reaches a far lower error than uniform FMM at the same DOF
    ref = _oracle_T("te")
    degree = 12                                 # DOF ~ 2*degree = 24
    e_pmm = abs(_pmm_T("te", degree) - ref)
    of, Rf, Tf = rcwa_efficiency_1d(**GOLD, polarization="te", n_orders=degree)
    e_fmm = abs(float(Tf.sum()) - ref)
    assert e_pmm < e_fmm


# ---------------------------------------------------------------------------
# energy conservation (lossless) + the convention check (absorbing slab)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm_lossless_conserves_energy(pol):
    diel = dict(GOLD, n_ridge=2.5 + 0j)
    o, R, T = pmm_efficiency_1d(**diel, polarization=pol, degree=16)
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-6


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_pmm_absorbing_slab_matches_fmm(pol):
    # ridge == groove == an absorbing medium -> a plain slab (no grating); PMM
    # and FMM must agree to machine precision (validates the public-convention
    # end-to-end: a mixed convention makes an absorber give R+T > 1).
    slab = dict(period=1.0e-6, n_ridge=1.5 + 0.3j, n_groove=1.5 + 0.3j,
                n_substrate=1.0, n_superstrate=1.0, depth=0.4e-6,
                duty_cycle=0.5, wavelength=0.8e-6)
    o, Rp, Tp = pmm_efficiency_1d(**slab, polarization=pol, degree=10)
    o2, Rf, Tf = rcwa_efficiency_1d(**slab, polarization=pol, n_orders=15)
    assert abs(float(Rp.sum()) - float(Rf.sum())) < 1e-9
    assert abs(float(Tp.sum()) - float(Tf.sum())) < 1e-9
    assert float(Rp.sum() + Tp.sum()) < 1.0     # genuine absorption, not > 1


# ---------------------------------------------------------------------------
# mesh grading (the TM speed lever) helps, and never breaks correctness
# ---------------------------------------------------------------------------

def test_pmm_grading_helps_metal_tm():
    ref = _oracle_T("tm")
    e_graded = abs(_pmm_T("tm", 8, elements_per_region=3, grade=True) - ref)
    e_uniform = abs(_pmm_T("tm", 8, elements_per_region=3, grade=False) - ref)
    assert e_graded < e_uniform                 # corner clustering helps


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def test_pmm_oblique_matches_fmm():
    # Oblique incidence (the +i*kx0 Bloch shift) -- once a NotImplementedError,
    # now solved: matches the FMM oracle for BOTH polarizations and conserves
    # energy.  Dielectric grating (the robust case; lossy-metal-TM at steep
    # oblique stays resonance-limited and is not asserted here).
    diel = dict(period=0.8e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.5,
                n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.5,
                wavelength=0.55e-6)
    ang = np.radians(25.0)
    for pol in ("te", "tm"):
        o, R, T = pmm_efficiency_1d(**diel, polarization=pol, degree=20,
                                    angle=ang)
        orf, Rr, Tr = rcwa_efficiency_1d(**diel, polarization=pol, n_orders=31,
                                         angle=ang)
        assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-4   # lossless
        dr = {int(a): float(b) for a, b in zip(orf, Rr)}
        dt = {int(a): float(b) for a, b in zip(orf, Tr)}
        pr = {int(a): float(b) for a, b in zip(o, R)}
        pt = {int(a): float(b) for a, b in zip(o, T)}
        keys = set(dr) | set(pr)
        err = max(max(abs(dr.get(k, 0) - pr.get(k, 0)) for k in keys),
                  max(abs(dt.get(k, 0) - pt.get(k, 0)) for k in keys))
        assert err < 1e-3, f"{pol}: PMM oblique vs FMM max|d|={err:.2e}"


def test_pmm_rejects_bad_polarization():
    with pytest.raises(ValueError, match="polarization must be"):
        pmm_efficiency_1d(**GOLD, polarization="tem", degree=10)


def test_pmm_rejects_low_degree():
    with pytest.raises(ValueError, match="degree must be"):
        pmm_efficiency_1d(**GOLD, polarization="te", degree=1)


def test_pmm_exported_at_top_level():
    import lumenairy as la
    assert hasattr(la, "pmm_efficiency_1d")

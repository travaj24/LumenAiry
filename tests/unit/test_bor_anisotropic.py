"""BOR anisotropy: diagonal CYLINDRICAL permittivity diag(eps_rr, eps_phiphi,
eps_zz) in the radial eigensolver and on ``BORStack.add_layer``.

The physics anchor is the uniaxial dispersion in a UNIFORM medium at m = 0,
where the cylindrical problem splits exactly:

  * TE0 (E_phi only)      -- ordinary:      q^2 + gamma^2 = k0^2 eps_phiphi
  * TM0 (E_r, E_z)        -- extraordinary: gamma^2/eps_zz + q^2/eps_rr = k0^2

gamma is fixed by the radial discretisation (grid + wall + m), NOT by eps, so
it can be read off an isotropic reference run and reused -- which makes the
comparison discretisation-independent.
"""
import numpy as np
import pytest

from lumenairy.elements.bor import BORStack
from lumenairy.elements.bor.coupled_radial_eigensolver import (
    radial_coupled_modes,
)

K0 = 2 * np.pi
RBIG, NRAD, M = 8.0, 100, 0
EO, EE = 2.25, 3.24            # the exp10/exp26 LC: n_o = 1.5, n_e = 1.8


def _uniform(tri):
    tri = np.asarray(tri, complex)
    if tri.size == 1:
        return lambda r: np.full(np.asarray(r).size, tri.item(), complex)
    return lambda r: np.repeat(tri[None, :], np.asarray(r).size, axis=0)


def _modes(tri):
    return radial_coupled_modes(M, RBIG, NRAD, _uniform(tri), K0, staggered=True)


def _split(mds):
    """Propagating forward modes, split into (TE, TM) by where the E-energy is."""
    p = [d for d in mds
         if abs(np.real(d["q"])) > 1e-6 and np.real(d["q"] ** 2) > 0]
    te, tm = [], []
    for d in p:
        e_te = np.sum(np.abs(d["Ephi"]) ** 2)
        e_tm = np.sum(np.abs(d["Er"]) ** 2) + np.sum(np.abs(d["Ez"]) ** 2)
        (te if e_te > e_tm else tm).append(complex(d["q"]))
    return (sorted(te, key=lambda z: -z.real), sorted(tm, key=lambda z: -z.real))


def test_isotropic_tensor_is_byte_identical_to_scalar():
    """diag(e, e, e) must reproduce the scalar path EXACTLY -- the anisotropy
    generalisation must not perturb any existing isotropic result."""
    q_scalar = np.sort_complex(np.array([d["q"] for d in _modes([EO])]))
    q_tensor = np.sort_complex(np.array([d["q"] for d in _modes([EO, EO, EO])]))
    assert np.abs(q_scalar - q_tensor).max() == 0.0


def test_te_ordinary_is_blind_to_eps_zz():
    """At m = 0 the TE branch carries only E_phi, so eps_zz cannot move it."""
    te_i, _ = _split(_modes([EO, EO, EO]))
    te_a, _ = _split(_modes([EO, EO, EE]))
    k = min(len(te_i), len(te_a))
    assert k >= 8
    assert max(abs(te_i[i] - te_a[i]) for i in range(k)) == 0.0


def test_tm_extraordinary_matches_uniaxial_dispersion():
    """TM branch must follow gamma^2/eps_zz + q^2/eps_rr = k0^2."""
    _, tm_i = _split(_modes([EO, EO, EO]))
    _, tm_a = _split(_modes([EO, EO, EE]))
    k = min(len(tm_i), len(tm_a))
    assert k >= 8
    worst = 0.0
    for i in range(k):
        g2 = K0 ** 2 * EO - tm_i[i] ** 2          # gamma^2 from the iso run
        pred = np.sqrt(EO * (K0 ** 2 - g2 / EE))
        worst = max(worst, abs(pred - tm_a[i]) / abs(pred))
    assert worst < 1e-12, f"extraordinary dispersion off by {worst:.2e}"


@pytest.mark.parametrize("tri", [[EE, EO, EO], [EO, EE, EO]])
def test_wrong_component_slot_is_rejected_by_the_oracle(tri):
    """Discriminating power: putting eps_e in the WRONG diagonal slot must blow
    the uniaxial prediction, so this suite cannot pass a slot-swap refactor."""
    _, tm_i = _split(_modes([EO, EO, EO]))
    _, tm_w = _split(_modes(tri))
    k = min(len(tm_i), len(tm_w))
    worst = 0.0
    for i in range(k):
        g2 = K0 ** 2 * EO - tm_i[i] ** 2
        pred = np.sqrt(EO * (K0 ** 2 - g2 / EE))
        worst = max(worst, abs(pred - tm_w[i]) / abs(pred))
    assert worst > 1e-2, "wrong-slot tensor still matched -- oracle is blind"


def test_borstack_anisotropic_layer_cascades_and_conserves_energy():
    s = BORStack(6.0, 1, n_substrate=1.0, n_superstrate=1.0, N=90)
    s.add_layer(0.4, eps=1.0)
    s.add_layer(0.5, eps_tensor=(EO, EO, EE))       # uniform uniaxial slab
    s.add_layer(0.4, eps=1.0)
    s.set_source(wavelength=1.0)
    res = s.solve()
    energy = np.atleast_1d(np.asarray(res["energy"], float))
    assert energy.size > 0
    assert np.max(np.abs(energy - 1.0)) < 1e-8


def test_borstack_anisotropic_profile_layer_runs():
    def tprof(r):
        r = np.asarray(r)
        core = r < 2.0
        out = np.empty((r.size, 3), complex)
        out[:, 0] = np.where(core, EO, 1.0)
        out[:, 1] = np.where(core, EO, 1.0)
        out[:, 2] = np.where(core, EE, 1.0)
        return out

    s = BORStack(6.0, 0, n_substrate=1.0, n_superstrate=1.0, N=90)
    s.add_layer(0.5, eps=1.0)
    s.add_layer(0.5, eps_tensor_profile=tprof)
    s.add_layer(0.5, eps=1.0)
    s.set_source(wavelength=1.0)
    res = s.solve()
    assert np.atleast_1d(np.asarray(res["R"], float)).size > 0


def test_add_layer_rejects_two_specs_and_bad_tensor_length():
    s = BORStack(4.0, 0, N=40)
    with pytest.raises(ValueError, match="EXACTLY ONE"):
        s.add_layer(0.1, eps=1.0, eps_tensor=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="3 DIAGONAL"):
        s.add_layer(0.1, eps_tensor=(1.0, 1.0))

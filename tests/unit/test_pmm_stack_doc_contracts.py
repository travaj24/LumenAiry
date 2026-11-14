"""Pins for two statements in the PMMStack docstrings.

* ``n_orders`` / ``far_field_orders`` set how many Rayleigh orders the far
  field projects; they do not control accuracy (the modal truncation is
  ``degree`` x ``elements_per_region``).
* The reference planes of the zeroth-order Jones: reflection at the top
  interface of the first layer, transmission at the bottom interface of the
  last layer, lab tangential ``(E_x, E_y)`` over incident, ``exp(-i w t)``.
"""
import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack

WL, P, W = 1300e-9, 370e-9, 175e-9
F = W / P
N_SUB = 1.447


def _fixture_a(n_orders):
    st = PMMStack(P, n_substrate=N_SUB, n_superstrate=1.0, degree=20,
                  elements_per_region=3, n_orders=n_orders)
    for t, es, eg in [(50e-9, -74 + 6j, 1.0), (1e-9, 3.05, 3.05),
                      (7e-9, 3.53, 3.53), (5e-9, 2.5 + 0.5j, 2.5 + 0.5j),
                      (7e-9, 3.53, 3.53), (1e-9, 3.05, 3.05),
                      (70e-9, -160 + 30j, -160 + 30j)]:
        if es == eg:
            st.add_layer(t, eps=es)
        else:
            st.add_layer(t, segments=[(F, es), (1 - F, eg)])
    st.set_source(WL)
    return st


def test_n_orders_is_not_an_accuracy_knob():
    """``n_orders`` 3 / 21 / 61 give the same zeroth-order Jones.  MEASURED
    2026-11-14: relative differences <= 4.3e-15 (J_xx) / 3.0e-16 (J_yy) on
    Windows/MKL and <= 3.8e-15 / 2.1e-15 on WSL/OpenBLAS."""
    J = [np.asarray(_fixture_a(n).solve()[3]) for n in (3, 21, 61)]
    for Jn in J[1:]:
        assert abs(Jn[0, 0] - J[0][0, 0]) < 1e-12 * abs(J[0][0, 0])
        assert abs(Jn[1, 1] - J[0][1, 1]) < 1e-12 * abs(J[0][1, 1])


# --------------------------------------------------------------------------- #
# reference planes
# --------------------------------------------------------------------------- #
EPS_FILM, T_FILM, D_PAD = 2.5 + 0.5j, 80e-9, 100e-9


def _film(theta, top_air=0.0, bottom_sub=0.0):
    st = PMMStack(P, n_substrate=N_SUB, n_superstrate=1.0, degree=8)
    if top_air:
        st.add_layer(top_air, eps=1.0)
    st.add_layer(T_FILM, eps=EPS_FILM)
    if bottom_sub:
        st.add_layer(bottom_sub, eps=N_SUB ** 2)
    st.set_source(WL, theta=theta)
    J = np.asarray(st.solve()[3])
    return J, np.asarray(st.jones_transmission())


def _closed_form_r(theta, pol):
    k0 = 2 * np.pi / WL
    kx = np.sin(theta)

    def kz(eps):
        v = np.sqrt(complex(eps - kx ** 2))
        return -v if v.imag < 0 else v

    def Y(eps):
        return kz(eps) if pol == "te" else eps / kz(eps)
    Ys, Yl, Y0 = Y(N_SUB ** 2), Y(EPS_FILM), Y(1.0)
    tn = np.tan(k0 * kz(EPS_FILM) * T_FILM)
    Yin = Yl * (Ys - 1j * Yl * tn) / (Yl - 1j * Ys * tn)
    # tangential-E reflection, admittance kz (TE) / eps / kz (TM): the
    # lab-basis Jones entries J[1, 1] / J[0, 0]
    return (Y0 - Yin) / (Y0 + Yin)


#: MEASURED 2026-11-14 (degree 8, both builds): closed form vs library
#: <= 3.7e-14; the plane-shift identities <= 4.1e-14 (4.05e-14 WSL); the
#: substrate pad moves reflection by <= 6.1e-17.  1e-9 is the plan's
#: statement-level bar; |r| ~ 0.27 and the phase factors are unimodular.
BAR = 1e-9


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_reflection_matches_closed_form(theta):
    J, _Jt = _film(theta)
    assert abs(J[1, 1] - _closed_form_r(theta, "te")) < BAR
    assert abs(J[0, 0] - _closed_form_r(theta, "tm")) < BAR


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_reference_planes(theta):
    k0 = 2 * np.pi / WL
    c = np.cos(theta)
    kz_sub = np.sqrt(N_SUB ** 2 - np.sin(theta) ** 2)
    J0, Jt0 = _film(theta)
    Ja, Jta = _film(theta, top_air=D_PAD)
    Jb, Jtb = _film(theta, bottom_sub=D_PAD)
    # reflection: referenced at the top interface of the first layer, so an
    # air layer on top adds the round trip exp(2 i k0 cos(theta) d)
    ph_r = np.exp(2j * k0 * c * D_PAD)
    for i in range(2):
        assert abs(Ja[i, i] - J0[i, i] * ph_r) < BAR
    # transmission: referenced at the bottom interface of the last layer
    ph_top = np.exp(1j * k0 * c * D_PAD)
    ph_bot = np.exp(1j * k0 * kz_sub * D_PAD)
    for i in range(2):
        assert abs(Jta[i, i] - Jt0[i, i] * ph_top) < BAR
        assert abs(Jtb[i, i] - Jt0[i, i] * ph_bot) < BAR
    # and the substrate-material pad leaves reflection unchanged
    assert float(np.max(np.abs(Jb - J0))) < BAR

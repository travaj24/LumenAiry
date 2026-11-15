"""The statements of ``RCWAResult.jones_reflection`` / ``jones_transmission``
docstrings, pinned.

* ``J`` is the ratio of zeroth-order TANGENTIAL lab fields (``E_x``, ``E_y``),
  public ``exp(-i w t)``: on a uniform film it equals the closed-form
  tangential-field (characteristic-matrix) coefficients, TE and TM;
* the REFLECTION phase is referenced at the top interface of the first layer:
  ``d`` of superstrate material added on top multiplies ``J_r`` by
  ``exp(2 i k0 cos(theta) d)``;
* the TRANSMISSION phase is referenced at the bottom interface of the last
  layer: ``d`` of substrate material added at the bottom multiplies ``J_t`` by
  ``exp(i k0 kz_sub d)``, and ``d`` of superstrate added on top by
  ``exp(i k0 cos(theta) d)``;
* at ``phi = 0``, ``J_r[0, 0] = -r_p`` (textbook Fresnel ``r_p``) and the TM
  transmission entry is ``t_p (kz_sub / n_sub) / (kz_sup / n_sup)``.

Fixture: an 80 nm film (eps 2.2+0.6i) on n = 1.5, 1500 nm, air above, at
theta 0 and 0.3 rad (a generic film; re-measured): every entry agrees with the
closed form to <= 3e-16 at n_orders = 3 (a uniform stack has no truncation
error).
"""
import numpy as np
import pytest

from lumenairy.elements.rcwa import RCWAStack

WL, NSUB, EPS_F, D_F, D_ADD = 1500e-9, 1.5, 2.2 + 0.6j, 80e-9, 100e-9
K0 = 2 * np.pi / WL
BAR = 1e-9


def _kz(eps, s):
    """Forward kz / k0 in a medium of permittivity ``eps`` (Im >= 0)."""
    k = np.sqrt(complex(eps) - s * s)
    return k if k.imag >= 0 else -k


def _closed_form(theta, pol, films):
    """Tangential-field coefficients ``(r, t)`` of a film stack (top first)
    between air and the substrate, ``exp(-i w t)``: ``r = E_t^r / E_t^i`` at
    the top interface, ``t = E_t^t / E_t^i`` at the bottom interface, with the
    TE admittance ``kz`` and the TM admittance ``eps / kz``."""
    s = np.sin(theta)

    def eta(eps):
        kz = _kz(eps, s)
        return kz if pol == "te" else eps / kz

    E, H = 1.0 + 0j, eta(NSUB ** 2)          # at the bottom interface
    for eps, d in reversed(films):
        delta = K0 * _kz(eps, s) * d
        e = eta(eps)
        E, H = (np.cos(delta) * E - 1j * np.sin(delta) / e * H,
                -1j * e * np.sin(delta) * E + np.cos(delta) * H)
    e0 = eta(1.0)
    r = (e0 * E - H) / (e0 * E + H)
    return r, 1.0 / (E / (1.0 + r))


def _rcwa(theta, films):
    st = RCWAStack(400e-9, n_superstrate=1.0, n_substrate=NSUB, n_orders=3)
    for eps, d in films:
        st.add_layer(d, eps=eps)
    res = st.set_source(WL, theta=theta).solve()
    return np.asarray(res.jones_reflection()), np.asarray(
        res.jones_transmission())


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_jones_is_the_closed_form_tangential_ratio(theta):
    Jr, Jt = _rcwa(theta, [(EPS_F, D_F)])
    for idx, pol in ((0, "tm"), (1, "te")):
        r, t = _closed_form(theta, pol, [(EPS_F, D_F)])
        assert abs(Jr[idx, idx] - r) < BAR
        assert abs(Jt[idx, idx] - t) < BAR
    # no cross-polarization on an isotropic film at phi = 0
    assert abs(Jr[0, 1]) < BAR and abs(Jr[1, 0]) < BAR


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_phase_reference_planes(theta):
    Jr, Jt = _rcwa(theta, [(EPS_F, D_F)])
    c = np.cos(theta)
    # superstrate (air) added on top
    Jr_a, Jt_a = _rcwa(theta, [(1.0, D_ADD), (EPS_F, D_F)])
    assert np.max(np.abs(Jr_a - Jr * np.exp(2j * K0 * c * D_ADD))) < BAR
    assert np.max(np.abs(Jt_a - Jt * np.exp(1j * K0 * c * D_ADD))) < BAR
    # substrate material added at the bottom: transmission only
    Jr_s, Jt_s = _rcwa(theta, [(EPS_F, D_F), (NSUB ** 2, D_ADD)])
    kz_sub = np.sqrt(NSUB ** 2 - np.sin(theta) ** 2)
    assert np.max(np.abs(Jr_s - Jr)) < BAR
    assert np.max(np.abs(Jt_s - Jt * np.exp(1j * K0 * kz_sub * D_ADD))) < BAR


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.9])
def test_p_channel_against_textbook_fresnel(theta):
    """A bare air / glass interface (a layer of substrate material): the
    ``E_x`` reflection is ``-r_p`` (textbook Fresnel convention; at normal
    incidence ``J_xx = J_yy``), the ``E_x`` transmission is
    ``t_p cos(theta_t) / cos(theta_i)``, i.e. ``t_p`` times the ratio of the
    media's ``kz / n``."""
    Jr, Jt = _rcwa(theta, [(NSUB ** 2, D_ADD)])
    kz_sub = np.sqrt(NSUB ** 2 - np.sin(theta) ** 2)
    ci, ct = np.cos(theta), kz_sub / NSUB
    rp = (NSUB * ci - ct) / (NSUB * ci + ct)
    tp = 2 * ci / (NSUB * ci + ct)
    rs = (ci - NSUB * ct) / (ci + NSUB * ct)
    ph = np.exp(1j * K0 * kz_sub * D_ADD)       # the added layer's own phase
    assert abs(Jr[0, 0] - (-rp)) < BAR
    assert abs(Jr[1, 1] - rs) < BAR
    assert abs(Jt[0, 0] - tp * (kz_sub / NSUB) / ci * ph) < BAR
    assert abs(Jt[1, 1] - (1 + rs) * ph) < BAR


def test_docstrings_state_the_reference_planes():
    from lumenairy.elements.rcwa.stack import RCWAResult
    dr = RCWAResult.jones_reflection.__doc__
    dt = RCWAResult.jones_transmission.__doc__
    assert "top interface of the FIRST layer" in dr
    assert "BOTTOM interface of the LAST layer" in dt
    for d in (dr, dt):
        assert "exp(-i w t)" in d and "TANGENTIAL" in d

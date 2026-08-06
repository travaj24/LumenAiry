# Verifier round 2 (2026-08-06), sibling of V3: the standoff resolver and
# near-focus bridge gate measured the beam about the GRID ORIGIN.  A beam
# decentred by x_c read sqrt(2 x_c^2 + w^2) -- 2.34x too wide at 1.5 w --
# which corrupted the waist estimate (w0 ~ 1/w_env, zR ~ 1/w_env^2) and
# resolved a 6.3x-SHORTER leg, shrinking the Bluestein period 6.3x.
# Fixed by measuring about the intensity centroid (sub-pixel snap keeps the
# on-axis universe byte-identical) and by taking the containment half-width
# to the NEAREST edge (grid half-width minus decentre).
import numpy as np

from lumenairy.propagators import carrier as C

LAM = 1.31e-6
DX = 2e-6
N = 512
W = 40e-6
R = -20e-3


def _leg(xc):
    g = (np.arange(N) - N / 2) * DX
    E = np.exp(-(((g[None, :] - xc) ** 2 + g[:, None] ** 2))
               / W ** 2).astype(complex)
    return C._default_focus_standoff(E, R, 20e-3, LAM, DX)


def test_centred_beam_centroid_snaps_to_exact_zero():
    g = (np.arange(N) - N / 2) * DX
    E = np.exp(-((g[None, :] ** 2 + g[:, None] ** 2)) / W ** 2).astype(complex)
    assert C._envelope_amp_centroid(E, DX, DX) == (0.0, 0.0)


def test_decentred_leg_follows_the_beam_not_the_origin():
    # Pre-fix the 1.5 w decentre resolved a ~6.3x SHORTER leg (period ~6.3x
    # shorter).  Post-fix the leg tracks the beam: mildly LONGER, because the
    # nearest edge is closer so the effective extent is smaller and the
    # containment law asks for a longer hand-off.  Comparative bar: within
    # 2x of on-axis (vs 6.3x off before), and on the long side.
    s0 = _leg(0.0)
    s15 = _leg(1.5 * W)
    assert s0 > 0.0
    assert 1.0 <= s15 / s0 < 2.0, s15 / s0


def test_more_decentre_means_longer_leg_monotone():
    # Less room to the near edge -> smaller effective extent -> longer leg.
    s15 = _leg(1.5 * W)
    s3 = _leg(3.0 * W)
    assert s3 > s15


def test_width_read_is_decentre_invariant():
    # The 2.34x-too-wide reading was the root cause; the centroid-referenced
    # width must be decentre-invariant to well under the old error.
    g = (np.arange(N) - N / 2) * DX
    E0 = np.exp(-((g[None, :] ** 2 + g[:, None] ** 2)) / W ** 2).astype(complex)
    Ec = np.exp(-(((g[None, :] - 1.5 * W) ** 2 + g[:, None] ** 2))
                / W ** 2).astype(complex)
    w0 = C._envelope_amp_radius(E0, DX, DX,
                                centre=C._envelope_amp_centroid(E0, DX, DX))
    wc = C._envelope_amp_radius(Ec, DX, DX,
                                centre=C._envelope_amp_centroid(Ec, DX, DX))
    assert abs(wc / w0 - 1.0) < 1e-3, wc / w0

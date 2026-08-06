# Verifier finding V3, MFT sibling (2026-08-06): the P11 output-window
# warning in ``lumenairy/propagators/mft.py`` was centre_out-blind -- the
# faithful zone of the discrete transform is centred on the transform
# ORIGIN, not on ``centre_out``, so an off-origin window spends the period
# budget at weight two (``2*|centre_out| + N_out*d_out <= period``).  The
# carrier-side readout guard got the geometry fix with V3 proper; this file
# pins the same condition on the three public MFT propagators' warning.
import warnings

import numpy as np
import pytest

import lumenairy as la

LAM = 633e-9
DX = 2e-6
N = 64
Z = 0.5e-3


def _gauss():
    g = (np.arange(N) - N / 2) * DX
    return np.exp(-(g[:, None] ** 2 + g[None, :] ** 2)
                  / (2 * (6e-6) ** 2)).astype(complex)


def _replica_warns(record):
    return [w for w in record if 'REPLICAS' in str(w.message)]


def _asm(centre, frac=0.8):
    period = N * DX
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        la.angular_spectrum_propagate_mft(
            _gauss(), Z, LAM, DX, frac * period / N, N, centre_out=centre)
    return _replica_warns(rec), period


def test_on_origin_window_inside_period_is_silent():
    warns, _period = _asm((0.0, 0.0))
    assert not warns


def test_off_origin_same_window_now_warns():
    # 2*|c| + W = (1.0 + 0.8) * period > period: replicas enter the window
    # even though the window alone fits.  This was silent before the fix.
    warns, _period = _asm((N * DX / 2, 0.0))
    assert len(warns) == 1
    msg = str(warns[0].message)
    assert 'faithful zone' in msg and 'weight two' in msg
    # the message must not resurrect the old false claim
    assert 'of centre_out are PERIODIC REPLICAS' not in msg


def test_offset_budget_is_weight_two():
    # Boundary: 2*|c| + W == period stays silent; one output pixel past it
    # warns.  The exchange rate between offset and window is exactly 1:2.
    period = N * DX
    frac = 0.5
    c_edge = (period - frac * period) / 2.0
    warns_at, _ = _asm((c_edge * (1 - 1e-6), 0.0), frac=frac)
    warns_past, _ = _asm((c_edge + frac * period / N, 0.0), frac=frac)
    assert not warns_at
    assert len(warns_past) == 1


@pytest.mark.parametrize('fn', ['fresnel_propagate_mft',
                                'fraunhofer_propagate_mft'])
def test_field_transform_kernels_share_the_condition(fn):
    period = LAM * Z / DX
    prop = getattr(la, fn)
    def run(centre):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            prop(_gauss(), Z, LAM, DX, 0.8 * period / N, N,
                 centre_out=centre)
        return _replica_warns(rec)
    assert not run((0.0, 0.0))
    assert len(run((period / 2, 0.0))) == 1

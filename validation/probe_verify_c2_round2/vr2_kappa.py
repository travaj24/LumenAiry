"""VERIFY-WP-C2 ROUND 2, item D2 -- an INDEPENDENT recomputation of the
conditioning number the restated ``_conditioning_bar`` now uses.

Everything is rebuilt here: this probe does not import the test's helper.
It builds the same public fixture, forms its OWN finite-difference Jacobian
of ``propagate_modal_asymptotic`` in the fit's phase coefficients, and
reduces it to the induced ``inf <- 2`` operator norm THREE independent ways:

* a closed-form per-row eigenvalue of the 2x2 Gram matrix of
  ``[Re J_row; Im J_row]`` (what the test does);
* a full ``numpy.linalg.svd`` of every row's 2 x n matrix (no Gram, no
  closed form) -- this is the arithmetic the closed form could get wrong;
* a POWER ITERATION over the whole operator (and a random-direction sample)
  as a LOWER bound, which is what catches a norm that is too small.

Three step sizes are used for the Jacobian so the finite difference is shown
converged rather than assumed, and the whole column set is re-derived for
each.  ``n`` (70 for this fixture) is reported together with
``fit.coef_phi.size`` so "is 70 columns enough" is answered by construction:
the Jacobian has one column PER COEFFICIENT, so it is the complete operator
on that input space, not a subsample.  The OTHER float inputs the field
reads are enumerated and their own sensitivities measured separately, which
is the part the claim does not cover.

Finally the 10x bar is bracketed from both sides on the running build: a
drift at ``0.9 x`` the bar must pass and one at ``1.1 x`` must fail.

Usage:  LUMENAIRY_ROOT=<root> python vr2_kappa.py <out.json>
"""
import dataclasses
import json
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))
from test_audit_propagation import _build_singlet_fit  # noqa: E402

from lumenairy.propagators.asymptotic import propagate_modal_asymptotic  # noqa: E402

EPS = float(np.finfo(np.float64).eps)


def fixture_kwargs():
    N = 32
    s2x = np.linspace(-5e-6, 5e-6, N)
    s2y = np.linspace(-5e-6, 5e-6, N)
    S2X, S2Y = np.meshgrid(s2x, s2y, indexing='xy')
    return dict(
        source_point=(0.0, 0.0),
        source_amplitudes={(0, 0): 1.0 + 0.0j},
        pupil_amplitudes={(0, 0): 1.0 + 0.0j},
        w_s=50e-6, w_p=0.02, v2_centre=(0.0, 0.0),
        s2_grid_x=S2X, s2_grid_y=S2Y,
    )


def jacobian(fit, kwargs, base_flat, peak, delta):
    coef = np.asarray(fit.coef_phi, dtype=float)
    cols = np.empty((base_flat.size, coef.size), dtype=np.complex128)
    for j in range(coef.size):
        bump = np.zeros_like(coef)
        bump[j] = delta
        alt = dataclasses.replace(fit, coef_phi=coef * (1.0 + bump))
        cols[:, j] = (
            np.asarray(propagate_modal_asymptotic(alt, **kwargs)).ravel()
            - base_flat) / (peak * delta)
    return cols


def kappa_closed_form(cols):
    re, im = cols.real, cols.imag
    aa = np.einsum('ij,ij->i', re, re)
    bb = np.einsum('ij,ij->i', im, im)
    ab = np.einsum('ij,ij->i', re, im)
    tr = aa + bb
    det = aa * bb - ab * ab
    lam = 0.5 * (tr + np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0)))
    w = int(np.argmax(lam))
    return float(np.sqrt(lam[w])), w


def kappa_full_svd(cols):
    """No Gram, no closed form: an SVD per row."""
    best, wrow, wdir = 0.0, -1, None
    for i in range(cols.shape[0]):
        m = np.stack([cols[i].real, cols[i].imag])
        _u, sv, vt = np.linalg.svd(m, full_matrices=False)
        if sv[0] > best:
            best, wrow, wdir = float(sv[0]), i, vt[0]
    return best, wrow, wdir


def kappa_power_iteration(cols, iters=200, seed=11):
    """Lower bound: maximise ||J v||_inf over ||v||_2 = 1 by iterating the
    per-row 2x2 operator of the currently-winning row."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=cols.shape[1])
    v /= np.linalg.norm(v)
    best = 0.0
    for _ in range(iters):
        resp = np.abs(cols @ v)
        i = int(np.argmax(resp))
        best = max(best, float(resp[i]))
        m = np.stack([cols[i].real, cols[i].imag])
        g = m.T @ (m @ v)
        nrm = np.linalg.norm(g)
        if nrm == 0.0:
            break
        v = g / nrm
    return best, v


def main(out_path):
    fit = _build_singlet_fit()
    kwargs = fixture_kwargs()
    coef = np.asarray(fit.coef_phi, dtype=float)
    base = np.asarray(propagate_modal_asymptotic(fit, **kwargs))
    base_flat = base.ravel()

    # peak: the test uses the COLD-START reference's peak; the field's own
    # peak is within a rounding of it and is what makes this probe
    # independent of the test's reference implementation.
    peak = float(np.max(np.abs(base)))

    out = {
        'lumenairy_file': la.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'n_coefficients': int(coef.size),
        'n_pixels': int(base_flat.size),
        'peak': peak,
        'eps': EPS,
    }

    per_delta = {}
    for delta in (1e-12, 1e-11, 1e-10):
        cols = jacobian(fit, kwargs, base_flat, peak, delta)
        k_cf, row_cf = kappa_closed_form(cols)
        k_svd, row_svd, direction = kappa_full_svd(cols)
        k_pow, v_pow = kappa_power_iteration(cols)
        rng = np.random.default_rng(20260920)
        d0 = rng.normal(size=coef.shape)
        d0 /= np.linalg.norm(d0)
        k_rand = float(np.max(np.abs(cols @ d0)))
        per_delta['%g' % delta] = {
            'kappa_closed_form': k_cf, 'winning_row_closed_form': row_cf,
            'kappa_full_svd': k_svd, 'winning_row_svd': row_svd,
            'kappa_power_iteration_lower_bound': k_pow,
            'kappa_along_shipped_random_direction': k_rand,
            'closed_form_vs_svd_rel': abs(k_cf - k_svd) / k_svd,
            'power_vs_svd_rel': abs(k_pow - k_svd) / k_svd,
        }
        if delta == 1e-11:
            dir_ref, kappa_ref = direction, k_svd
    out['per_delta'] = per_delta
    ks = [v['kappa_full_svd'] for v in per_delta.values()]
    out['kappa'] = float(kappa_ref)
    out['kappa_spread_over_fd_steps'] = max(ks) / min(ks)

    # the NONLINEAR check the test makes: a real move along the attaining
    # direction reproduces the linearised worst case
    ladder = {}
    for delta in (1e-12, 1e-11, 1e-10):
        alt = dataclasses.replace(fit, coef_phi=coef * (1.0 + delta * dir_ref))
        moved = np.asarray(propagate_modal_asymptotic(alt, **kwargs))
        ladder['%g' % delta] = float(
            np.max(np.abs(moved - base))) / peak / delta
    out['response_along_attaining_direction'] = ladder
    lv = list(ladder.values())
    out['ladder_spread'] = max(lv) / min(lv)
    out['ladder_vs_kappa'] = max(lv) / kappa_ref

    # ---- is 70 columns the whole input space? -------------------------
    # coef_phi is the ONLY array the bar perturbs.  Enumerate the fit's
    # other float fields so the scope of the claim is a list rather than an
    # assumption, and measure the field's response to a relative move of
    # each of them along a random direction, for comparison.
    others = {}
    for f in dataclasses.fields(fit):
        if f.name == 'coef_phi':
            continue
        v = getattr(fit, f.name)
        arr = np.asarray(v) if not isinstance(v, (str, bool)) else None
        if arr is None or arr.dtype.kind not in 'fc' or arr.size == 0:
            others[f.name] = {'kind': type(v).__name__, 'perturbable': False}
            continue
        rng = np.random.default_rng(4242)
        d = rng.normal(size=arr.shape)
        d /= np.linalg.norm(d)
        try:
            alt = dataclasses.replace(fit, **{f.name: arr * (1.0 + 1e-11 * d)})
            moved = np.asarray(propagate_modal_asymptotic(alt, **kwargs))
            resp = float(np.max(np.abs(moved - base))) / peak / 1e-11
        except Exception as exc:                       # noqa: BLE001
            resp = 'ERROR: %s' % type(exc).__name__
        others[f.name] = {'kind': type(v).__name__, 'size': int(arr.size),
                          'perturbable': True,
                          'response_to_random_relative_move': resp}
    out['other_float_inputs'] = others

    # ---- the bar, bracketed ------------------------------------------
    floor = EPS * kappa_ref
    bar = 10.0 * floor
    out['floor_eps_times_kappa'] = floor
    out['bar_10x_floor'] = bar

    def _drift(scale):
        """A drift engineered to land at ``scale`` x the bar, measured."""
        delta = scale * bar / kappa_ref
        alt = dataclasses.replace(fit, coef_phi=coef * (1.0 + delta * dir_ref))
        moved = np.asarray(propagate_modal_asymptotic(alt, **kwargs))
        return float(np.max(np.abs(moved - base))) / peak

    brack = {}
    for scale in (0.5, 0.9, 1.1, 2.0, 100.0):
        m = _drift(scale)
        brack['%g' % scale] = {'measured_over_bar': m / bar,
                               'passes_10x_bar': bool(m <= bar)}
    out['bracket'] = brack

    # and what the bar would have been with the SHIPPED SEED's kappa
    k_seed = per_delta['1e-11']['kappa_along_shipped_random_direction']
    out['kappa_ratio_true_over_seeded'] = kappa_ref / k_seed

    with open(out_path, 'w') as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        if k in ('per_delta', 'other_float_inputs', 'bracket',
                 'response_along_attaining_direction'):
            print(k + ':')
            for kk, vv in v.items():
                print('   %-22s %s' % (kk, vv))
        else:
            print('%-40s %s' % (k, v))
    print('wrote', out_path)


if __name__ == '__main__':
    main(sys.argv[1])

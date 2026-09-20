"""VERIFY-WP-B12b probe W5 -- WHICH operation makes
``propagate_gbd_through_prescription`` differ from ``apply_real_lens_gbd`` in
its last bits on one build and not on the other, and whether the reconstructed
field's exact bytes depend on the memory budget.

WP-B12b records the difference ("byte-identical on Windows, not on WSL, while
the fidelity agrees to fifteen digits") without saying what causes it.  This
probe walks the two routes stage by stage on one fixture:

1. the INPUT each route decomposes -- ``apply_real_lens_gbd`` clips the
   entrance aperture first, the propagators-level entry does not;
2. the BUNDLE each route evolves -- ``apply_real_lens_gbd`` additionally
   prunes the zero-amplitude beamlets (``_prune_zero_beamlets``), which
   changes both the COUNT and the ORDER of the terms in the coherent sum;
3. the FIELD -- reconstructed from the same evolved bundle with the two
   routes' own reconstruction arguments.

Then the order-sensitivity control: the SAME bundle reconstructed with
different ``chunk_beamlets`` and ``LUMENAIRY_MEM_BUDGET_MB``, which changes
only the grouping of a floating-point sum.

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import inspect
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vb12b_common import (  # noqa: E402
    assert_tree,
    dump,
    env_block,
    fidelity,
    fixtures,
    sha,
)

FRAME = dict(sample_step=4, waist_factor=4.0)


def _bsha(b):
    return sha(np.concatenate([
        np.asarray(b.positions).ravel(), np.asarray(b.directions).ravel(),
        np.asarray(b.Q).ravel().view(np.float64),
        np.asarray(b.amplitude).ravel().view(np.float64),
        np.asarray(b.waist0).ravel()]))[:24]


def main():
    assert_tree()
    import lumenairy as la
    from lumenairy.elements.lenses_gbd import _prune_zero_beamlets
    from lumenairy.propagators import gbd as G
    arm = ('pre' if '_Rl = float(' in inspect.getsource(
        G.apply_prescription_persurface_to_beamlets) else 'post')
    fx = fixtures()['flatbase_asph']
    # A REDUCED grid: this probe is about which STAGE differs, not about
    # accuracy, and the per-surface path is ~N^4 here.
    fx.N, fx.dx, fx.semi, fx.w0 = 112, 3.6e-6, 0.17e-3, 0.105e-3
    zf = fx.best_focus()
    E = fx.E_in()
    out = dict(env=env_block(), arm=arm, fixture=fx.key, z=float(zf),
               frame=FRAME)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = np.asarray(la.apply_real_lens_gbd(
            E, prescription=fx.prescription(), wavelength=fx.lam, dx=fx.dx,
            output_plane_distance=float(zf), **FRAME))
        p = np.asarray(G.propagate_gbd_through_prescription(
            E, fx.dx, fx.prescription(), wavelength=fx.lam,
            output_shape=(fx.N, fx.N), output_dx=fx.dx, per_surface=True,
            z_image=float(zf), **FRAME))
        u = np.asarray(la.apply_real_lens_universal(
            E, prescription=fx.prescription(), wavelength=fx.lam, dx=fx.dx,
            method='gbd', output_plane_distance=float(zf),
            method_kwargs={'gbd': dict(FRAME)}))

        # stage 1: the input each route decomposes
        X, Y = fx.grid()
        E_clip = np.where(np.hypot(X, Y) <= fx.semi, E, 0.0)
        # stage 2: the bundles
        b_raw = G.decompose_field_to_beamlets(E, fx.dx, wavelength=fx.lam,
                                              **FRAME)
        b_clip = G.decompose_field_to_beamlets(E_clip, fx.dx,
                                               wavelength=fx.lam, **FRAME)
        b_pruned, n_kept = _prune_zero_beamlets(b_clip)
        ev_raw = G.apply_prescription_persurface_to_beamlets(
            b_raw, fx.prescription(), fx.lam, z_image=float(zf))
        ev_pruned = G.apply_prescription_persurface_to_beamlets(
            b_pruned, fx.prescription(), fx.lam, z_image=float(zf))
        f_raw = np.asarray(G.reconstruct_field_from_beamlets(
            ev_raw, Ny=fx.N, Nx=fx.N, dx=fx.dx, wavelength=fx.lam,
            window=5.0))
        f_pruned = np.asarray(G.reconstruct_field_from_beamlets(
            ev_pruned, Ny=fx.N, Nx=fx.N, dx=fx.dx, wavelength=fx.lam,
            window=5.0))
        # stage 3: order sensitivity of the coherent sum
        chunked = {}
        for ch in (2048, 512, 97):
            chunked[ch] = sha(np.asarray(G.reconstruct_field_from_beamlets(
                ev_pruned, Ny=fx.N, Nx=fx.N, dx=fx.dx, wavelength=fx.lam,
                window=5.0, chunk_beamlets=ch)))[:24]
        budgets = {}
        for mb in (4096.0, 512.0, 1.0):
            budgets[mb] = sha(np.asarray(G.reconstruct_field_from_beamlets(
                ev_pruned, Ny=fx.N, Nx=fx.N, dx=fx.dx, wavelength=fx.lam,
                window=5.0, mem_budget_mb=mb)))[:24]

    out.update(
        sha_apply_real_lens_gbd=sha(a)[:24],
        sha_propagate=sha(p)[:24], sha_universal=sha(u)[:24],
        universal_is_gbd_bytes=bool(sha(u) == sha(a)),
        propagate_is_gbd_bytes=bool(sha(p) == sha(a)),
        propagate_vs_gbd_fidelity=fidelity(p, a),
        propagate_vs_gbd_max_abs=float(np.abs(p - a).max()),
        propagate_vs_gbd_rel=float(np.abs(p - a).max()
                                   / max(np.abs(a).max(), 1e-300)),
        n_beamlets_raw=int(np.asarray(b_raw.positions).shape[0]),
        n_beamlets_clipped=int(np.asarray(b_clip.positions).shape[0]),
        n_beamlets_pruned=int(n_kept),
        sha_bundle_raw=_bsha(b_raw), sha_bundle_pruned=_bsha(b_pruned),
        n_evolved_raw=int(np.asarray(ev_raw.positions).shape[0]),
        n_evolved_pruned=int(np.asarray(ev_pruned.positions).shape[0]),
        sha_field_from_raw=sha(f_raw)[:24],
        sha_field_from_pruned=sha(f_pruned)[:24],
        raw_vs_pruned_max_abs=float(np.abs(f_raw - f_pruned).max()),
        raw_vs_pruned_rel=float(np.abs(f_raw - f_pruned).max()
                                / max(np.abs(f_pruned).max(), 1e-300)),
        field_from_pruned_is_gbd=bool(sha(f_pruned) == sha(a)),
        field_from_raw_is_propagate=bool(sha(f_raw) == sha(p)),
        chunk_digests=chunked, budget_digests=budgets,
        chunk_digests_agree=bool(len(set(chunked.values())) == 1),
        budget_digests_agree=bool(len(set(budgets.values())) == 1),
    )
    for k, v in out.items():
        if k != 'env':
            print(f'{k:32s} {v}')
    dump(out, f'probe_w5_entries_{arm}')


if __name__ == '__main__':
    main()

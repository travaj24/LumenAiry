"""Audit v5.17.0 Wave 5 regression pins: propagators cluster (P2-26..P2-32).

- P2-26  real-dtype E_in into ``angular_spectrum_propagate`` is cast to the
         target complex dtype BEFORE ``_fft2`` -- no pyFFTW rejection, no
         shape blacklisting, no misleading 'memory pressure' warning.
- P2-27  ``angular_spectrum_propagate_batch`` fetches H through the shared
         ``_get_asm_H_natural`` helper instead of running a full wasted
         FFT+IFFT pair on a garbage proxy field; batch output still equals
         per-component scalar calls.
- P2-28  ``angular_spectrum_propagate_tilted`` caches H in NATURAL layout
         and uses the exact 2-shift fold; output is bit-identical to the
         old 4-shift centered-H idiom.
- P2-29  fresnel/fraunhofer complex64 carrier argument is accumulated in
         float64 (mft-style f64-carrier-then-cast); complex128 unchanged.
- P2-30  ``apply_abcd_to_beamlets`` amplitude uses the Collins/Siegman
         factor 1/(A + B*Q_in) instead of Q_new/Q_old (which carried a
         spurious (C*q_in + D)); composite-ABCD now matches the sequential
         per-leg path and the analytic Gaussian-beam formulas.
- P2-31  ``propagate_hfpi``'s normalization warning states exactly which
         Fresnel-Kirchhoff factors are and are not applied (doc pin).
- P2-32  HFPI end-to-end helpers promote a REAL input dtype to the
         matching complex dtype before ``accumulate_to_grid`` so the
         imaginary half of the path weights is not silently discarded.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy.propagators.fft_infra as fi
from lumenairy.propagators import asm
from lumenairy.propagators.asm import (
    _build_asm_H_square,
    _get_asm_H_natural,
    angular_spectrum_propagate,
    angular_spectrum_propagate_tilted,
)
from lumenairy.propagators.fresnel import fraunhofer_propagate, fresnel_propagate
from lumenairy.propagators.gbd import (
    BeamletBundle,
    apply_abcd_to_beamlets,
    apply_thin_lens_to_beamlets,
    propagate_beamlets_freespace,
)
from lumenairy.propagators.hfpi import (
    _complex_output_dtype,
    propagate_hfpi,
    propagate_hfpi_freespace_aperture,
)

# Access the batch entry point as a module attribute (it is deliberately
# not in ``asm.__all__``; see the V9-walker note there).
angular_spectrum_propagate_batch = asm.angular_spectrum_propagate_batch


# ---------------------------------------------------------------------------
# P2-26: real-dtype input must not poison the pyFFTW shape blacklist
# ---------------------------------------------------------------------------

class TestP226RealInputCast:

    def test_real_input_equals_precast_complex(self):
        """A real E_in propagates identically to casting it complex first."""
        E_real = np.ones((64, 64), dtype=np.float64)
        target = np.dtype(fi.DEFAULT_COMPLEX_DTYPE)
        out_real = angular_spectrum_propagate(E_real, 1e-3, 633e-9, 5e-6)
        out_cplx = angular_spectrum_propagate(
            E_real.astype(target), 1e-3, 633e-9, 5e-6)
        assert out_real.dtype == target
        assert np.array_equal(out_real, out_cplx)

    @pytest.mark.skipif(not fi.PYFFTW_AVAILABLE, reason="pyFFTW not installed")
    def test_real_input_does_not_blacklist_shape(self):
        """Pre-fix: a real float64 field at N >= FFTW_MIN_SIZE hit pyFFTW
        uncast, raised ValueError internally, emitted a RuntimeWarning
        blaming 'memory pressure', and permanently blacklisted the SHAPE
        for all dtypes (every later complex call at that shape silently
        ran on scipy)."""
        N = int(fi.FFTW_MIN_SIZE)
        shape = (N, N)
        saved = set(fi._PYFFTW_BAD_SHAPES)
        fi._PYFFTW_BAD_SHAPES.discard(shape)
        try:
            E = np.ones(shape, dtype=np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                out = angular_spectrum_propagate(E, 1e-3, 633e-9, 5e-6)
            assert np.iscomplexobj(out)
            assert shape not in fi._PYFFTW_BAD_SHAPES
        finally:
            fi._PYFFTW_BAD_SHAPES.clear()
            fi._PYFFTW_BAD_SHAPES.update(saved)


# ---------------------------------------------------------------------------
# P2-27: batch must not run a wasted scalar FFT+IFFT pair on a proxy field
# ---------------------------------------------------------------------------

class TestP227BatchNoProxy:

    def _stack(self, N=64, cdtype=np.complex128):
        rng = np.random.default_rng(0)
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(cdtype)
        return np.stack([E, 1j * E])

    def test_no_scalar_fft_pair_per_batch_call(self, monkeypatch):
        """One warm batch call must dispatch exactly one ND FFT pair and
        ZERO scalar 2-D FFTs (the pre-fix proxy round-trip)."""
        stack = self._stack()
        z, wl, dx = 1e-3, 633e-9, 5e-6
        angular_spectrum_propagate_batch(stack, z, wl, dx)  # warm H cache

        counts = {}
        for name in ('_fft2', '_ifft2', '_fft2_nd', '_ifft2_nd'):
            orig = getattr(asm, name)

            def _wrap(*a, __n=name, __f=orig, **k):
                counts[__n] = counts.get(__n, 0) + 1
                return __f(*a, **k)

            counts[name] = 0
            monkeypatch.setattr(asm, name, _wrap)

        angular_spectrum_propagate_batch(stack, z, wl, dx)
        assert counts['_fft2'] == 0, "proxy scalar FFT pair is back (P2-27)"
        assert counts['_ifft2'] == 0
        assert counts['_fft2_nd'] == 1
        assert counts['_ifft2_nd'] == 1

    @pytest.mark.parametrize('cdtype', [np.complex128, np.complex64])
    def test_batch_equals_scalar_calls(self, cdtype):
        stack = self._stack(cdtype=cdtype)
        z, wl, dx = 1e-3, 633e-9, 5e-6
        out_b = angular_spectrum_propagate_batch(stack, z, wl, dx)
        out_s = np.stack([
            angular_spectrum_propagate(stack[0], z, wl, dx),
            angular_spectrum_propagate(stack[1], z, wl, dx),
        ])
        assert out_b.dtype == cdtype
        tol = 1e-12 if cdtype == np.complex128 else 1e-5
        scale = np.max(np.abs(out_s))
        assert np.max(np.abs(out_b - out_s)) <= tol * scale

    def test_helper_matches_square_reference(self):
        """fftshift(_get_asm_H_natural(...)) reproduces the documented
        centered single-source-of-truth ``_build_asm_H_square``."""
        N, dx, z, wl = 64, 5e-6, 1e-3, 633e-9
        H_nat = _get_asm_H_natural(N, N, dx, dx, wl, z, True,
                                   np.dtype(np.complex128), np)
        H_ref = _build_asm_H_square(N, dx, z, wl, dtype=np.complex128,
                                    bandlimit=True)
        assert np.array_equal(np.fft.fftshift(H_nat), H_ref)


# ---------------------------------------------------------------------------
# P2-28: tilted ASM 2-shift natural-H fold, bit-identical to 4-shift idiom
# ---------------------------------------------------------------------------

class TestP228TiltedFold:

    @pytest.mark.parametrize('N,cdtype', [(64, np.complex128),
                                          (65, np.complex128),
                                          (64, np.complex64)])
    def test_output_bit_identical_to_4shift_centered_idiom(self, N, cdtype):
        rng = np.random.default_rng(7)
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(cdtype)
        z, wl, dx = 2e-3, 1.55e-6, 1e-6
        tx, ty = 0.02, -0.013
        out = angular_spectrum_propagate_tilted(E, z, wl, dx,
                                                tilt_x=tx, tilt_y=ty)
        assert out.dtype == cdtype

        # Recover the cached H (now NATURAL layout) via the exact key.
        fx0 = np.sin(tx) / wl
        fy0 = np.sin(ty) / wl
        key = (N, N, float(dx), float(dx), float(wl), float(z),
               float(fx0), float(fy0), True, np.dtype(cdtype).str,
               'ASM_TILTED')
        H_nat = fi._h_cache_lookup(key)
        assert H_nat is not None, "tilted H not cached under expected key"

        # Rebuild the carrier exactly as the function does.
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        ph = (-2.0 * np.pi) * (fx0 * X + fy0 * Y)
        if np.dtype(cdtype) == np.complex64:
            p = np.mod(ph, 2.0 * np.pi)
            carrier = np.empty(p.shape, dtype=np.complex64)
            carrier.real[:] = np.cos(p).astype(np.float32)
            carrier.imag[:] = np.sin(p).astype(np.float32)
        else:
            carrier = np.exp(1j * ph)
        E_demod = E.astype(cdtype, copy=False) * carrier

        # Historical 4-shift centered-H idiom on the re-centered H.
        H_centered = np.fft.fftshift(H_nat)
        E_fft = np.fft.fftshift(fi._fft2(np.fft.ifftshift(E_demod)))
        E_prop = np.fft.fftshift(
            fi._ifft2(np.fft.ifftshift(E_fft * H_centered)))
        out_4shift = E_prop * np.conj(carrier)

        assert np.array_equal(out, out_4shift), \
            "2-shift natural-H fold is not bit-identical to 4-shift idiom"


# ---------------------------------------------------------------------------
# P2-29: c64 fresnel/fraunhofer carrier must be accumulated at float64
# ---------------------------------------------------------------------------

class TestP229CarrierPrecision:

    def test_fresnel_c64_tracks_c128_reference(self):
        """Annular edge-aperture field (carrier error is worst at the grid
        edge).  Pre-fix (f32 carrier): rel error 1.7e-5 at these params;
        f64-carrier-then-cast: 3.5e-7."""
        N, dx, z, wl = 256, 4e-6, 5e-3, 1.55e-6
        x = (np.arange(N, dtype=np.float64) - N / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        r2 = X**2 + Y**2
        E = np.where((r2 > (0.7 * N / 2 * dx) ** 2)
                     & (r2 < (0.95 * N / 2 * dx) ** 2), 1.0, 0.0)
        ref, _, _ = fresnel_propagate(E.astype(np.complex128), z, wl, dx)
        o64, _, _ = fresnel_propagate(E.astype(np.complex64), z, wl, dx)
        assert o64.dtype == np.complex64
        rel = np.max(np.abs(o64 - ref)) / np.max(np.abs(ref))
        assert rel < 3e-6, f"c64 Fresnel carrier drift {rel:.3e} (P2-29)"

    def test_fraunhofer_c64_tracks_c128_reference(self):
        """Random-phase field spreads energy across the whole output plane
        so the output-prefactor carrier error is not masked.  Pre-fix:
        4.3e-5; post-fix: 2.0e-7."""
        N, dx, z, wl = 256, 8e-6, 10e-3, 1.55e-6
        rng = np.random.default_rng(5)
        E = np.exp(1j * 2 * np.pi * rng.random((N, N)))
        ref, _, _ = fraunhofer_propagate(E.astype(np.complex128), z, wl, dx)
        o64, _, _ = fraunhofer_propagate(E.astype(np.complex64), z, wl, dx)
        assert o64.dtype == np.complex64
        rel = np.max(np.abs(o64 - ref)) / np.max(np.abs(ref))
        assert rel < 3e-6, f"c64 Fraunhofer carrier drift {rel:.3e} (P2-29)"


# ---------------------------------------------------------------------------
# P2-30: apply_abcd_to_beamlets must use the Collins factor 1/(A + B*Q_in)
# ---------------------------------------------------------------------------

def _one_beamlet(wl=1e-6, w0=5e-6):
    z_R = np.pi * w0**2 / wl
    return BeamletBundle(
        positions=np.array([[0.0, 0.0, 0.0]]),
        directions=np.array([[0.0, 0.0, 1.0]]),
        Q=np.array([-1j / z_R], dtype=np.complex128),
        amplitude=np.array([1.0 + 0j]),
        waist0=np.array([w0]),
    )


class TestP230CollinsFactor:
    wl = 1e-6
    w0 = 5e-6

    def test_composite_abcd_matches_sequential_legs(self):
        """t1 -> thin lens f -> t2: the single-ABCD path must reproduce
        the sequential per-leg path (which composes exactly to the
        analytic Collins result exp(ikL)/(A + B*Q0)).  Pre-fix the
        ABCD amplitude was low by (C*q0 + D) = 0.600 - 0.0016j here."""
        t1, f, t2 = 20e-3, 50e-3, 30e-3
        M = (np.array([[1, t2], [0, 1]], float)
             @ np.array([[1, 0], [-1 / f, 1]], float)
             @ np.array([[1, t1], [0, 1]], float))
        A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        L = t1 + t2

        b_seq = _one_beamlet(self.wl, self.w0)
        b_seq = propagate_beamlets_freespace(b_seq, t1, self.wl)
        b_seq = apply_thin_lens_to_beamlets(b_seq, f, self.wl)
        b_seq = propagate_beamlets_freespace(b_seq, t2, self.wl)

        b_abcd = apply_abcd_to_beamlets(_one_beamlet(self.wl, self.w0),
                                        A, B, C, D, self.wl, axial_opl=L)

        assert abs(b_abcd.Q[0] - b_seq.Q[0]) < 1e-12 * abs(b_seq.Q[0])
        ratio = b_abcd.amplitude[0] / b_seq.amplitude[0]
        assert abs(ratio - 1.0) < 1e-9, (
            f"composite-ABCD amplitude off by {ratio} vs sequential legs "
            f"(pre-fix this ratio was the spurious Collins excess C*q0+D)")

    def test_matches_analytic_collins_and_gaussian_w_of_z(self):
        """Free-space ABCD: amplitude factor 1/(1 + z*Q0); |factor| must
        equal the analytic w0/w(z)."""
        z = 7e-3
        b0 = _one_beamlet(self.wl, self.w0)
        Q0 = b0.Q[0]
        out = apply_abcd_to_beamlets(b0, 1.0, z, 0.0, 1.0, self.wl,
                                     axial_opl=z)
        k = 2 * np.pi / self.wl
        expect = np.exp(1j * k * z) / (1.0 + z * Q0)
        assert abs(out.amplitude[0] - expect) < 1e-12 * abs(expect)
        z_R = np.pi * self.w0**2 / self.wl
        w_z = self.w0 * np.sqrt(1.0 + (z / z_R) ** 2)
        assert abs(abs(1.0 / (1.0 + z * Q0)) - self.w0 / w_z) < 1e-12

    def test_thin_lens_abcd_amplitude_preserved(self):
        """B = 0 (bare thin lens): Collins factor is exactly 1, matching
        ``apply_thin_lens_to_beamlets`` for an on-axis beamlet.  Pre-fix
        the ABCD path multiplied by (C*q_in + D) != 1."""
        f = 50e-3
        b_abcd = apply_abcd_to_beamlets(_one_beamlet(self.wl, self.w0),
                                        1.0, 0.0, -1.0 / f, 1.0, self.wl)
        b_lens = apply_thin_lens_to_beamlets(_one_beamlet(self.wl, self.w0),
                                             f, self.wl)
        assert abs(b_abcd.amplitude[0] - b_lens.amplitude[0]) < 1e-14
        assert abs(b_abcd.Q[0] - b_lens.Q[0]) < 1e-12 * abs(b_lens.Q[0])


# ---------------------------------------------------------------------------
# P2-31: propagate_hfpi warning text must match the code (doc pin)
# ---------------------------------------------------------------------------

class TestP231WarningText:

    def test_warning_states_applied_and_missing_factors(self):
        doc = propagate_hfpi.__doc__
        # The applied factors are affirmatively listed as applied.
        assert '**does** apply' in doc
        assert 'Kirchhoff prefactor -- at the source init' in doc
        # The genuinely-missing terms are listed as NOT applied.
        assert 'does **not** apply' in doc
        assert '1/r' in doc and 'geometric-spreading' in doc

    def test_old_false_claims_are_gone(self):
        doc = propagate_hfpi.__doc__
        # Pre-fix text claimed 1/(jlambda) and the MC solid-angle weight
        # were NOT applied (they are, since v4.10/v4.11.2) ...
        assert 'carries three normalization factors that this code does '\
               '**not**' not in doc
        # ... and guaranteed relative-intensity ratios, which the missing
        # per-path 1/r and binning Jacobian quantitatively break.
        assert 'relative-intensity ratios within a single experiment' not in doc


# ---------------------------------------------------------------------------
# P2-32: real-dtype E_in must not discard the imaginary half of the field
# ---------------------------------------------------------------------------

class TestP232RealInputComplexOutput:

    def test_complex_output_dtype_helper(self):
        assert _complex_output_dtype(np.float64) == np.complex128
        assert _complex_output_dtype(np.float32) == np.complex64
        assert _complex_output_dtype(np.complex128) == np.complex128
        assert _complex_output_dtype(np.complex64) == np.complex64
        assert _complex_output_dtype(np.int32) == np.complex128

    def test_real_input_keeps_imaginary_part(self):
        kw = dict(z_to_aperture=50e-6, aperture_radius=20e-6,
                  z_aperture_to_output=150e-6, wavelength=0.5e-6,
                  n_paths=20000, rng=42,
                  # v5.31 (audit W9-14): the new sampling-adequacy guard fires
                  # on this geometry -- of 20000 paths only ~114 land on the
                  # 32x32 grid (0.11 per pixel) because the default
                  # cone_half_angle is a full forward hemisphere.  That is a
                  # TRUE positive, but this test pins the real-input DTYPE
                  # contract, not sampling, and it runs under
                  # ``simplefilter("error")`` -- so acknowledge the guard here
                  # rather than let an unrelated diagnostic fail the dtype pin.
                  on_undersampled='silent')
        E_real = np.ones((32, 32), dtype=np.float64)
        with warnings.catch_warnings():
            # Pre-fix this emitted np.exceptions.ComplexWarning at the
            # scatter-add and returned Re(E) only.
            warnings.simplefilter("error")
            out_r = propagate_hfpi_freespace_aperture(E_real, 1e-6, **kw)
        out_c = propagate_hfpi_freespace_aperture(
            E_real.astype(np.complex128), 1e-6, **kw)
        assert out_r.dtype == np.complex128
        assert np.array_equal(out_r, out_c)
        assert np.max(np.abs(out_r.imag)) > 0.0

    def test_float32_input_promotes_to_complex64(self):
        kw = dict(z_to_aperture=50e-6, aperture_radius=20e-6,
                  z_aperture_to_output=150e-6, wavelength=0.5e-6,
                  n_paths=500, rng=1)
        out = propagate_hfpi_freespace_aperture(
            np.ones((8, 8), dtype=np.float32), 1e-6, **kw)
        assert out.dtype == np.complex64

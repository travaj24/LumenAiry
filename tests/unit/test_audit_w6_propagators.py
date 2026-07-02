"""Audit v5.17.0 Wave 6 regression pins: propagators cluster (P3 batch).

- P3-51  ``apply_fresnel_curvature`` honours dtype-follows-input: the
         f64-accumulated phase factor is cast to E's complex dtype before
         the multiply, so complex64 stays complex64 for R != 0 (matching
         the R == 0 early-return and the docstring); complex128 is
         byte-identical to pre-fix.
- P3-52  x64 policy: the four asymptotic JAX twins RAISE (rcwa-style
         ``_require_jax_x64``) when ``jax_enable_x64`` is off, instead of
         ``fit_canonical_polynomials_jax`` mutating the global config
         MID-CALL (unsafe inside jax.jit) while the other three silently
         degraded to single precision.
- P3-53  ``decompose_lg`` / ``decompose_hg`` 2-D meshgrid branch derives
         dx/dy from the axis that actually varies, so an
         ``indexing='ij'`` meshgrid gives the SAME coefficients as
         ``indexing='xy'`` (pre-fix: silent all-zero coefficients);
         sheared/ambiguous grids raise ValueError.
- P3-56  ``fresnel_propagate`` uses ``astype(..., copy=False)``
         (no avoidable full-grid copies); output byte-identical, and the
         no-copy path must never mutate the caller's E_in.
- P3-57  ``propagate_huygens_fresnel_with_opl_callable``'s dead
         ``chunk_output`` parameter is deprecated: default (None) is
         silent, an explicit value emits DeprecationWarning, results
         unchanged either way.
- W5 P2-26 hardening: the ``_fft2`` / ``_ifft2`` / ``_fft2_nd`` /
         ``_ifft2_nd`` pyFFTW gates themselves reject non-complex input
         (route to scipy/numpy), so no FUTURE real-input caller can
         poison the bare-shape ``_PYFFTW_BAD_SHAPES`` blacklist.
"""
from __future__ import annotations

import subprocess
import sys
import warnings

import numpy as np
import pytest

import lumenairy.propagators.fft_infra as fi
from lumenairy.propagators.asm import apply_fresnel_curvature
from lumenairy.propagators.asymptotic_modes import decompose_hg, decompose_lg
from lumenairy.propagators.fresnel import fresnel_propagate
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_with_opl_callable,
)

try:
    import jax  # noqa: F401
    _HAS_JAX = True
except ImportError:                  # pragma: no cover - env dependent
    _HAS_JAX = False


# ---------------------------------------------------------------------------
# P3-51: apply_fresnel_curvature dtype-follows-input
# ---------------------------------------------------------------------------

class TestP351CurvatureDtype:

    def test_complex64_stays_complex64(self):
        """Pre-fix: R != 0 silently promoted c64 -> c128 (while R == 0
        returned c64), contradicting the docstring's 'same dtype'."""
        E = np.ones((64, 64), np.complex64)
        out = apply_fresnel_curvature(E, 1e-6, 1.31e-6, R=0.05)
        assert out.dtype == np.complex64
        # ... and it must agree with the R=0 branch's dtype contract.
        out0 = apply_fresnel_curvature(E, 1e-6, 1.31e-6, R=0.0)
        assert out.dtype == out0.dtype

    def test_complex128_bit_identical_to_direct_formula(self):
        """The c128 path must remain the plain f64 formula (the cast is
        a no-op there)."""
        rng = np.random.default_rng(0)
        E = rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))
        dx, wl, R = 1e-6, 1.31e-6, 0.05
        out = apply_fresnel_curvature(E, dx, wl, R=R)
        n = 32
        ax = (np.arange(n) - n / 2) * dx
        Y, X = np.meshgrid(ax, ax, indexing='ij')
        k = 2.0 * np.pi / wl
        ref = E * np.exp(1j * k * (X * X + Y * Y) / (2.0 * R))
        assert np.array_equal(out.view(np.uint8), ref.view(np.uint8))

    def test_complex64_values_match_f64_carrier(self):
        """The carrier must still be accumulated at f64 -- the c64 result
        equals the c128 result to c64 rounding."""
        rng = np.random.default_rng(1)
        E128 = rng.standard_normal((48, 48)) + 1j * rng.standard_normal((48, 48))
        E64 = E128.astype(np.complex64)
        out128 = apply_fresnel_curvature(E128, 2e-6, 1.31e-6, R=0.01)
        out64 = apply_fresnel_curvature(E64, 2e-6, 1.31e-6, R=0.01)
        np.testing.assert_allclose(out64, out128.astype(np.complex64),
                                   rtol=2e-6, atol=2e-6)

    def test_real_input_promotes_to_complex128(self):
        """Real (non-complex) E keeps the historical c128 output."""
        E = np.ones((16, 16), np.float64)
        out = apply_fresnel_curvature(E, 1e-6, 1.31e-6, R=0.05)
        assert out.dtype == np.complex128


# ---------------------------------------------------------------------------
# P3-52: asymptotic JAX twins require x64 (no mid-call global mutation)
# ---------------------------------------------------------------------------

_P352_SUBPROCESS_SRC = r"""
import jax
assert not jax.config.jax_enable_x64, "expected x64 OFF by default"
from lumenairy.propagators.asymptotic_jax_twin import (
    aberration_tensor_lg00_jax,
    fit_canonical_polynomials_jax,
    propagate_modal_asymptotic_lg00_jax,
    solve_envelope_stationary_jax_ift,
)
n_raised = 0
for fn, args in [
    (aberration_tensor_lg00_jax, (None, (0.0, 0.0), (0.0, 0.0))),
    (propagate_modal_asymptotic_lg00_jax, (None, None, None, None)),
    (solve_envelope_stationary_jax_ift, (None, (0.0, 0.0), (0.0, 0.0))),
    (fit_canonical_polynomials_jax, ({}, 1.31e-6)),
]:
    try:
        if fn is solve_envelope_stationary_jax_ift:
            fn(*args, w_s=1e-5, w_p=0.05)
        else:
            fn(*args)
    except RuntimeError as e:
        assert 'jax_enable_x64' in str(e), str(e)
        n_raised += 1
assert n_raised == 4, n_raised
# THE P3-52 pin: the global config must NOT have been mutated mid-call.
assert not jax.config.jax_enable_x64, "global jax_enable_x64 was mutated!"
print('OK')
"""


class TestP352X64Policy:

    @pytest.mark.skipif(not _HAS_JAX, reason="could not import 'jax'")
    def test_twins_raise_without_x64_and_do_not_mutate_global(self):
        """With x64 OFF (subprocess -- the ambient test session may have
        x64 on), all four twins raise RuntimeError naming
        jax_enable_x64, and the global flag is UNCHANGED afterwards
        (pre-fix: fit_canonical_polynomials_jax warned and flipped the
        global config mid-call)."""
        import os

        import lumenairy
        repo_root = os.path.dirname(os.path.dirname(lumenairy.__file__))
        proc = subprocess.run(
            [sys.executable, "-c", _P352_SUBPROCESS_SRC],
            capture_output=True, text=True, timeout=300, cwd=repo_root)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().endswith("OK")

    @pytest.mark.skipif(not _HAS_JAX, reason="could not import 'jax'")
    def test_require_helper_passes_when_x64_enabled(self):
        import jax as _jax
        if not _jax.config.jax_enable_x64:
            pytest.skip("ambient session has x64 off")
        from lumenairy.propagators.asymptotic_jax_twin import _require_jax_x64
        _require_jax_x64("wave6-pin")  # must not raise


# ---------------------------------------------------------------------------
# P3-53: decompose_lg/hg must handle indexing='ij' meshgrids
# ---------------------------------------------------------------------------

class TestP353MeshgridOrientation:

    def _grids(self, n=48, pitch=4e-6):
        ax = (np.arange(n) - n / 2) * pitch
        return ax

    def test_lg_ij_equals_xy(self):
        """Pre-fix: an indexing='ij' meshgrid gave dx = 0, da = 0 and
        silently ALL-ZERO coefficients."""
        ax = self._grids()
        w = 60e-6
        Xij, Yij = np.meshgrid(ax, ax, indexing='ij')
        Xxy, Yxy = np.meshgrid(ax, ax, indexing='xy')
        f_ij = np.exp(-(Xij**2 + Yij**2) / w**2).astype(np.complex128)
        f_xy = np.exp(-(Xxy**2 + Yxy**2) / w**2).astype(np.complex128)
        c_ij = decompose_lg(f_ij, Xij, Yij, w, 2, 2)
        c_xy = decompose_lg(f_xy, Xxy, Yxy, w, 2, 2)
        assert abs(c_ij[(0, 0)]) > 0.0, "all-zero ij coefficients (P3-53)"
        for key in c_xy:
            np.testing.assert_allclose(c_ij[key], c_xy[key],
                                       rtol=1e-12, atol=1e-15)

    def test_hg_ij_equals_xy(self):
        ax = self._grids()
        w = 60e-6
        Xij, Yij = np.meshgrid(ax, ax, indexing='ij')
        Xxy, Yxy = np.meshgrid(ax, ax, indexing='xy')
        f_ij = np.exp(-(Xij**2 + Yij**2) / w**2).astype(np.complex128)
        f_xy = np.exp(-(Xxy**2 + Yxy**2) / w**2).astype(np.complex128)
        c_ij = decompose_hg(f_ij, Xij, Yij, w, w, 2, 2)
        c_xy = decompose_hg(f_xy, Xxy, Yxy, w, w, 2, 2)
        assert abs(c_ij[(0, 0)]) > 0.0
        for key in c_xy:
            np.testing.assert_allclose(c_ij[key], c_xy[key],
                                       rtol=1e-12, atol=1e-15)

    def test_xy_path_values_unchanged(self):
        """The 'xy' branch must keep the exact pre-fix step idiom
        (mean of the first diff column)."""
        ax = self._grids()
        w = 60e-6
        Xxy, Yxy = np.meshgrid(ax, ax, indexing='xy')
        f = np.exp(-(Xxy**2 + Yxy**2) / w**2).astype(np.complex128)
        c_2d = decompose_lg(f, Xxy, Yxy, w, 1, 1)
        c_1d = decompose_lg(f, ax, ax, w, 1, 1)
        for key in c_1d:
            np.testing.assert_allclose(c_2d[key], c_1d[key],
                                       rtol=1e-12, atol=1e-15)

    def test_sheared_grid_raises(self):
        ax = self._grids(n=16)
        Xij, Yij = np.meshgrid(ax, ax, indexing='ij')
        f = np.ones((16, 16), np.complex128)
        with pytest.raises(ValueError, match="could not infer the grid step"):
            decompose_lg(f, Xij + Yij, Yij, 60e-6, 1, 1)
        with pytest.raises(ValueError, match="could not infer the grid step"):
            decompose_hg(f, Xij + Yij, Yij, 60e-6, 60e-6, 1, 1)


# ---------------------------------------------------------------------------
# P3-56: fresnel_propagate no-copy astype must not alias-mutate E_in
# ---------------------------------------------------------------------------

class TestP356FresnelNoCopy:

    def test_input_not_mutated_and_output_correct(self):
        """copy=False makes E_in participate un-copied in the multiply --
        pin that the caller's buffer is untouched and the result equals
        the straightforward reference evaluation."""
        rng = np.random.default_rng(2)
        N, dx, wl, z = 64, 2e-6, 1.31e-6, 0.01
        E = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        E_before = E.copy()
        out, dxo, dyo = fresnel_propagate(E, z, wl, dx)
        assert np.array_equal(E.view(np.uint8), E_before.view(np.uint8)), \
            "fresnel_propagate mutated its input (P3-56 no-copy hazard)"
        # Reference: same formula, straight numpy.
        k = 2 * np.pi / wl
        x1 = (np.arange(N, dtype=np.float64) - N / 2) * dx
        X1, Y1 = np.meshgrid(x1, x1, indexing='xy')
        dx_out = wl * z / (N * dx)
        x2 = (np.arange(N, dtype=np.float64) - N / 2) * dx_out
        X2, Y2 = np.meshgrid(x2, x2, indexing='xy')
        E_mod = E * np.exp(1j * k / (2 * z) * (X1**2 + Y1**2))
        E_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_mod)))
        pref = (np.exp(1j * k * z) / (1j * wl * z)
                * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2)) * dx * dx)
        np.testing.assert_allclose(out, pref * E_fft, rtol=1e-10, atol=1e-12)
        assert dxo == pytest.approx(dx_out)

    def test_complex64_dtype_preserved(self):
        rng = np.random.default_rng(3)
        E = (rng.standard_normal((64, 64))
             + 1j * rng.standard_normal((64, 64))).astype(np.complex64)
        out, _, _ = fresnel_propagate(E, 0.01, 1.31e-6, 2e-6)
        assert out.dtype == np.complex64


# ---------------------------------------------------------------------------
# P3-57: chunk_output is deprecated dead code
# ---------------------------------------------------------------------------

class TestP357ChunkOutputDeprecated:

    def _run(self, chunk=None, record=None):
        def opl_fn(S1X, S1Y, s2x, s2y):
            if record is not None:
                record["n"] += 1
            return ((S1X - s2x)**2 + (S1Y - s2y)**2) / (2 * 0.01) / 1.31e-6

        E = np.ones((8, 8), np.complex128)
        g = (np.arange(4) - 2) * 1e-5
        kwargs = dict(opl_fn=opl_fn, output_grid_x=g, output_grid_y=g,
                      input_grid_dx=1e-6, wavelength=1.31e-6)
        if chunk is not None:
            kwargs["chunk_output"] = chunk
        return propagate_huygens_fresnel_with_opl_callable(E, **kwargs)

    def test_default_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            self._run()

    def test_explicit_value_warns_and_result_unchanged(self):
        out_default = self._run()
        with pytest.warns(DeprecationWarning, match="chunk_output"):
            out_chunk = self._run(chunk=7)
        assert np.array_equal(out_default.view(np.uint8),
                              out_chunk.view(np.uint8))

    def test_evaluation_count_is_per_pixel(self):
        """Sanity pin from the audit: evaluation is strictly per output
        pixel -- 17 opl_fn calls per pixel with Van Vleck on."""
        rec = {"n": 0}
        self._run(record=rec)
        assert rec["n"] == 16 * 17  # 4x4 output grid, 1 + 16 FD calls each


# ---------------------------------------------------------------------------
# W5 P2-26 hardening: the _fft2/_ifft2 gates reject non-complex input
# ---------------------------------------------------------------------------

class TestFftGateComplexOnly:

    @pytest.fixture(autouse=True)
    def _preserve_blacklist(self):
        saved = set(fi._PYFFTW_BAD_SHAPES)
        yield
        fi._PYFFTW_BAD_SHAPES.clear()
        fi._PYFFTW_BAD_SHAPES.update(saved)

    @pytest.mark.parametrize("fn,ref,shape", [
        (fi._fft2, np.fft.fft2, None),
        (fi._ifft2, np.fft.ifft2, None),
        (fi._fft2_nd, lambda a: np.fft.fft2(a, axes=(-2, -1)), (2,)),
        (fi._ifft2_nd, lambda a: np.fft.ifft2(a, axes=(-2, -1)), (2,)),
    ])
    def test_real_input_no_blacklist_and_correct(self, fn, ref, shape):
        """Pre-hardening: a real f64 array at N >= FFTW_MIN_SIZE reached
        pyFFTW, failed, warned about 'memory pressure', and blacklisted
        the shape for ALL dtypes.  Post: routes to scipy/numpy silently
        with the correct result and an untouched blacklist."""
        N = int(fi.FFTW_MIN_SIZE)
        full = ((N, N) if shape is None else shape + (N, N))
        fi._PYFFTW_BAD_SHAPES.discard(full)
        x = np.random.default_rng(4).standard_normal(full)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            y = fn(x)
        assert full not in fi._PYFFTW_BAD_SHAPES
        np.testing.assert_allclose(y, ref(x), rtol=1e-10, atol=1e-10)

    @pytest.mark.skipif(not fi.PYFFTW_AVAILABLE, reason="pyFFTW not installed")
    def test_complex_after_real_still_correct(self):
        """A complex call at the same shape right after a real call must
        produce the correct transform (pre-hardening it silently ran on
        scipy via the poisoned blacklist -- correct but slow; the pin
        here is that the blacklist stays clean)."""
        N = int(fi.FFTW_MIN_SIZE)
        fi._PYFFTW_BAD_SHAPES.discard((N, N))
        rng = np.random.default_rng(5)
        fi._fft2(rng.standard_normal((N, N)))          # real: fallback path
        assert (N, N) not in fi._PYFFTW_BAD_SHAPES
        xc = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
        yc = fi._fft2(xc)
        np.testing.assert_allclose(yc, np.fft.fft2(xc), rtol=1e-9, atol=1e-9)

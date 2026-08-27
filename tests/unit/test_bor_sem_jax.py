"""Differentiable (JAX) twin of the SEM-basis BOR stack solve (``_jax_sem``).

``BORStack(basis='sem').solve()`` now runs under a trace when any segment /
uniform-layer eps or layer thickness is a JAX array (walls, meshes and
half-spaces stay concrete).  Permittivity enters the SEM Galerkin blocks only
as three per-element scalars on geometry-only mass matrices, so the traced
part is the eps-weighted assembly, the E_z Schur elimination, the
equilibrated-fold custom-VJP eigensolve, the modal reconstruction, the
cross-tested mortar and the Redheffer cascade.

Mesh topology is value-dependent on this basis (the wavelength-resolution cap
sizes elements from the local |n|), so a traced eps needs
``BORStack(n_mesh_cap=...)`` -- a concrete UPPER bound on |n| that pins the
mesh across AD/FD evaluations (over-resolving is safe).  A concrete jnp value
needs no hint and reproduces the NumPy mesh byte-for-byte (the parity gates).

These tests pin: forward total-parity vs NumPy on a wall-bearing stack whose
traced solve goes through the CROSS-MESH mortar (2 of 3 interfaces), the same
for a diagonally ANISOTROPIC stack (transfers the NumPy slot oracle to the
twin) plus an adversarial rr<->phiphi slot swap, gradients vs central FD for
every traced input form (segment Re eps / Im eps / eps_rr triple component /
uniform eps= / thickness), the masked-array energy closure, and the
instructive no-hint error.  Validated at build: parity 6e-15..7e-14,
gradients 7e-7..4e-10 relative, slot swap moves sum(R) by 1.3e-2 (12 orders
above the parity floor).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.bor.bor_stack import BORStack

_WL = 2 * np.pi / 2.0                       # k0 = 2.0 (validated BOR scale)


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def _base(cap=None):
    return BORStack(3.0, 1, n_superstrate=1.4142, n_substrate=1.4142,
                    basis="sem", degree=6, n_mesh_cap=cap)


def _solve_segments(er, thk=0.5, cap=None):
    """Two wall-bearing segment layers -- the traced solve of this stack runs
    the cross-tested mortar on 2 of its 3 interfaces (measured)."""
    s = _base(cap)
    s.add_layer(thk, segments=[(0.8, (er, er, er)), (1.6, 2.0), (3.0, 2.0)])
    s.add_layer(0.3, segments=[(1.2, 5.0), (3.0, 2.0)])
    s.set_source(wavelength=_WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def test_sem_jax_forward_parity_total():
    """Traced sum(R)/sum(T) reproduce the NumPy SEM solve through the
    cross-mesh mortar interfaces (order-/gauge-invariant totals)."""
    _jax()
    import jax.numpy as jnp
    rc, rj = _solve_segments(6.0), _solve_segments(jnp.asarray(6.0 + 0j))
    assert "inc_mask" in rj                       # actually took the twin
    dR = abs(float(np.sum(rc["R"])) - float(np.sum(np.asarray(rj["R"]))))
    dT = abs(float(np.sum(rc["T"])) - float(np.sum(np.asarray(rj["T"]))))
    assert dR < 1e-9 and dT < 1e-9, f"|dR|={dR:.2e} |dT|={dT:.2e}"


def _solve_aniso(tri, jnp=None):
    s = _base()
    if jnp is not None:
        tri = tuple(jnp.asarray(complex(t)) for t in tri)
    s.add_layer(0.5, segments=[(0.8, tri), (3.0, 2.0)])
    s.add_layer(0.3, segments=[(1.2, (2.9, 3.1, 3.0)), (3.0, 2.0)])
    s.set_source(wavelength=_WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def test_sem_jax_anisotropic_parity_and_slot_swap():
    """Diagonal-anisotropy parity vs the slot-verified NumPy SEM (7.7e-15
    against the analytic uniaxial oracle), plus the adversarial probe: an
    rr<->phiphi swap must MOVE the answer, not vanish into the parity floor."""
    _jax()
    import jax.numpy as jnp
    tri = (6.2, 5.5, 5.8)
    rc, rj = _solve_aniso(tri), _solve_aniso(tri, jnp)
    dR = abs(float(np.sum(rc["R"])) - float(np.sum(np.asarray(rj["R"]))))
    assert dR < 1e-9, f"aniso parity |dR|={dR:.2e}"
    r_sw = _solve_aniso((5.5, 6.2, 5.8), jnp)
    move = abs(float(np.sum(np.asarray(r_sw["R"])))
               - float(np.sum(np.asarray(rj["R"]))))
    assert move > 1e-4, f"slot swap moved sum(R) by only {move:.2e}"


def test_sem_jax_segment_eps_gradient_matches_fd():
    """d sum(R) / d Re(segment eps) vs central FD (hint-pinned mesh)."""
    jax = _jax()
    import jax.numpy as jnp

    def loss(x):
        return jnp.sum(jnp.asarray(_solve_segments(x + 0j, cap=2.6)["R"]))

    g = float(jax.grad(loss)(jnp.asarray(6.0)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(6.0 + h)))
          - float(loss(jnp.asarray(6.0 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9, \
        f"AD={g:.6e} FD={fd:.6e}"


def test_sem_jax_thickness_gradient_matches_fd():
    """d sum(T) / d(layer thickness) vs central FD (traced phase)."""
    jax = _jax()
    import jax.numpy as jnp

    def loss(thk):
        return jnp.sum(jnp.asarray(
            _solve_segments(6.0 + 0j, thk=thk, cap=2.6)["T"]))

    g = float(jax.grad(loss)(jnp.asarray(0.5)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(0.5 + h)))
          - float(loss(jnp.asarray(0.5 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9, \
        f"AD={g:.6e} FD={fd:.6e}"


def test_sem_jax_aniso_component_gradient_matches_fd():
    """d sum(R) / d eps_rr alone (the LC-director design knob) vs FD."""
    jax = _jax()
    import jax.numpy as jnp

    def loss(err):
        s = _base(cap=2.6)
        s.add_layer(0.5, segments=[(0.8, (err + 0j, 5.5 + 0j, 5.8 + 0j)),
                                   (3.0, 2.0)])
        s.set_source(wavelength=_WL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jnp.sum(jnp.asarray(s.solve()["R"]))

    g = float(jax.grad(loss)(jnp.asarray(6.2)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(6.2 + h)))
          - float(loss(jnp.asarray(6.2 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9, \
        f"AD={g:.6e} FD={fd:.6e}"


def test_sem_jax_lossy_and_uniform_eps_gradients_match_fd():
    """d sum(T) / d Im(segment eps) (lossy) and d sum(R) / d(uniform eps=)
    (the sidecar path) vs central FD."""
    jax = _jax()
    import jax.numpy as jnp
    h = 1e-6

    def loss_im(ei):
        s = _base(cap=2.6)
        s.add_layer(0.5, segments=[(0.8, 6.0 + 1j * ei), (3.0, 2.0)])
        s.set_source(wavelength=_WL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jnp.sum(jnp.asarray(s.solve()["T"]))

    g = float(jax.grad(loss_im)(jnp.asarray(0.1)))
    fd = (float(loss_im(jnp.asarray(0.1 + h)))
          - float(loss_im(jnp.asarray(0.1 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9, \
        f"Im-eps AD={g:.6e} FD={fd:.6e}"

    def loss_u(x):
        s = _base(cap=2.6)
        s.add_layer(0.5, eps=x + 0j)
        s.set_source(wavelength=_WL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return jnp.sum(jnp.asarray(s.solve()["R"]))

    g = float(jax.grad(loss_u)(jnp.asarray(6.0)))
    fd = (float(loss_u(jnp.asarray(6.0 + h)))
          - float(loss_u(jnp.asarray(6.0 - h)))) / (2 * h)
    assert abs(g - fd) <= 1e-5 * max(abs(fd), 1.0) + 1e-9, \
        f"eps= AD={g:.6e} FD={fd:.6e}"


def test_sem_jax_energy_closure_lossless_masked():
    """The masked full-array R/T close energy per incident order (lossless)."""
    _jax()
    import jax.numpy as jnp
    rj = _solve_segments(jnp.asarray(6.0 + 0j))    # all-real -> lossless
    energy = np.asarray(rj["energy"])
    inc = np.asarray(rj["inc_mask"])
    per_order = energy[inc > 0.5]
    assert per_order.size >= 3
    assert np.max(np.abs(per_order - 1.0)) < 1e-6, \
        f"max|R+T-1|={np.max(np.abs(per_order - 1.0)):.2e}"


def test_sem_jax_traced_without_hint_raises_instructive():
    """A traced eps with no n_mesh_cap must fail with the actionable message,
    not a bare TracerArrayConversionError from inside the mesh builder."""
    jax = _jax()
    import jax.numpy as jnp

    def loss(x):
        s = _base()                                # no cap
        s.add_layer(0.5, segments=[(0.8, x + 0j), (3.0, 2.0)])
        s.set_source(wavelength=_WL)
        return jnp.sum(jnp.asarray(s.solve()["R"]))

    with pytest.raises(NotImplementedError, match="n_mesh_cap"):
        jax.grad(loss)(jnp.asarray(6.0))

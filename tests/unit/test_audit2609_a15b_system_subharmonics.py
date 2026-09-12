"""WP-A15b -- ``propagate_through_system`` forwards ``subharmonics``.

WP-A8 added Lane subharmonic levels to
:func:`lumenairy.generate_turbulence_screen` (audit section 5, E3) and noted in
its report (section 5 item 3) that the element-chain call site was the one
caller that could not reach the new knob: ``system.py``'s ``'turbulence'``
branch passed ``r0`` / ``L0`` / ``l0`` / ``seed`` and stopped there, so a chain
element dict carrying ``'subharmonics': 3`` was silently discarded -- the
"knob quietly dropped" class the audit catalogues in section 15.3.

The two properties that matter are opposite-facing and both are pinned here:

1. **Bit-identity of the default.**  ``subharmonics`` defaults to ``0`` in the
   generator, so an element dict WITHOUT the key must produce the same field,
   bit for bit, as before the passthrough existed.  The oracle is the generator
   called directly at ``subharmonics=0`` and applied by hand -- an independent
   path through the same maths -- compared with ``array_equal``, not a
   tolerance: a defaulted kwarg has no numerical content.
2. **The knob actually arrives.**  With ``'subharmonics': 3`` the chain must
   produce the screen the generator produces at 3 levels, and that screen must
   DIFFER from the 0-level one (otherwise the passthrough would be
   unfalsifiable -- the V1 defect of this audit).

FAIL-BEFORE: on the pre-fix tree property 2's ``array_equal`` against the
3-level screen fails and the chain instead matches the 0-level screen, because
the key never reached the generator.  MEASURED on the fixture below
(2026-09-12): the 3-level screen differs from the 0-level one by
max |dphi| = 2.6701 rad, rms 1.3526 rad (1 level: 0.8413 / 0.3906 rad), i.e.
the discarded knob is a many-radian effect, not a rounding one.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.elements import generate_turbulence_screen
from lumenairy.propagators.system import propagate_through_system

# A deliberately small, fully deterministic fixture: the screen is seeded, the
# chain has exactly one element, and the input is a unit plane wave, so the
# output field IS ``exp(1j * screen)`` and nothing else can contribute.
_N = 32
_DX = 5e-4          # 0.5 mm -- 16 mm aperture at N = 32
_WL = 1.31e-6
_R0 = 1e-2          # 10 mm Fried parameter: strong, so the screen is not tiny
_SEED = 20260912


def _plane_wave():
    return np.ones((_N, _N), dtype=np.complex128)


def _chain(extra):
    elem = {'type': 'turbulence', 'r0': _R0, 'seed': _SEED}
    elem.update(extra)
    E_out, _log = propagate_through_system(
        _plane_wave(), [elem], wavelength=_WL, dx=_DX)
    return np.asarray(E_out)


def _screen(subharmonics):
    return generate_turbulence_screen(
        _N, _DX, r0=_R0, L0=np.inf, l0=0.0, seed=_SEED,
        subharmonics=subharmonics)


def test_default_chain_is_bit_identical_to_the_zero_level_screen():
    """Property 1 -- the passthrough changes nothing when the key is absent."""
    got = _chain({})
    want = _plane_wave() * np.exp(1j * _screen(0))
    assert np.array_equal(got, want), (
        'the turbulence chain branch no longer reproduces the shipped '
        'subharmonics=0 screen bit for bit')


def test_explicit_zero_matches_the_absent_key_bit_for_bit():
    """``elem.get('subharmonics', 0)`` -- the default and an explicit 0 are
    the same call, which is what makes property 1 a real default rather than
    a coincidence."""
    assert np.array_equal(_chain({}), _chain({'subharmonics': 0}))


@pytest.mark.parametrize('levels', [1, 3])
def test_the_key_reaches_the_generator(levels):
    """Property 2 -- the chain's screen IS the generator's screen at the
    requested number of levels."""
    got = _chain({'subharmonics': levels})
    want = _plane_wave() * np.exp(1j * _screen(levels))
    assert np.array_equal(got, want), (
        f'subharmonics={levels} did not reach generate_turbulence_screen '
        f'through the element chain')


def test_the_knob_is_falsifiable_on_this_fixture():
    """Guard against the V1 shape (a pin that passes for any implementation).

    The two screens must differ by MUCH more than float64 noise, or property
    2 above would hold even if the key were still dropped.  Bar: 0.1 rad.
    Derivation: the Lane levels add power below the FFT lattice's fundamental
    ``1/(N dx) = 62.5 m^-1``, whose Kolmogorov PSD amplitude at this r0 is
    order 1 rad; the MEASURED difference on this fixture (2026-09-12) is
    max |dphi| = 2.6701 rad / rms 1.3526 rad, so the bar sits 27x below the
    signal and ~1e15 above the float64 floor of a ~1 rad quantity.
    """
    d = _screen(3) - _screen(0)
    assert np.max(np.abs(d)) > 0.1, (
        f'subharmonics=3 barely changes the screen on this fixture '
        f'(max |dphi| = {np.max(np.abs(d)):.3e} rad); the passthrough pins '
        f'above would not discriminate.  Re-choose r0 / N / dx.')


def test_the_element_docstring_documents_the_key():
    """audit section 15.3: a knob the chain accepts must be documented where
    a chain author looks for it, or it is an undiscoverable option."""
    assert 'subharmonics' in (propagate_through_system.__doc__ or '')

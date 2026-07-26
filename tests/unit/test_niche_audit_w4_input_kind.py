"""Audit A-9 (wave W4) -- close the deferred ``input_kind`` rollout.

`lumenairy/_validation.py::_check_2d_scalar_field` grew an
``input_kind`` kwarg in v4.15.5 (P2-NEW-F1-3) so the rejection message
could say "expected 2-D complex **pupil**" / "**psf**" instead of the
hardcoded "field".  It was then wired at **2 of 68** call sites (both in
``propagators/vector_diffraction.py``), and ``compute_psf`` was left
carrying a marker comment -- "``input_kind='pupil'`` would be ideal once
Agent B's parameterised ``_check_2d_scalar_field`` lands; the default
form here is correct in the interim" -- that outlived the thing it was
waiting for by five minor releases.  The audit deferred the rollout
twice as "wants its own AST-driven batch".

This file pins the closure:

1. Every call site declares an **explicit** ``input_kind`` (no reliance
   on the default), and the declared value matches the semantics of the
   argument being guarded.
2. A junk ``input_kind`` raises ``ValueError`` naming the offending
   value *and* the allowed set (house rule for enum-valued knobs),
   checked per wired site.
3. The vocabulary is closed (``_INPUT_KINDS``) rather than prose-only.
4. Structural fail-closed pin: **no** ``_check_2d_scalar_field`` call
   anywhere in ``lumenairy/`` may omit ``input_kind``, so the next entry
   point cannot land unwired (the 'fix N, miss N+1' meta-pattern).
5. Bit-identity spot checks: wiring changes only the *message wording*
   for rejected inputs, never the value returned for accepted ones.

**Rollout history: 2 -> 27 -> 68 of 68.**  v4.15.5 wired 2 (both in
``propagators/vector_diffraction.py``).  ce0a265 (v5.31) wired 25 more
and left a 41-site **handoff table** (``_HANDOFF_PREFIXES``) for the
modules then owned by sibling agents -- ``analysis/detector.py``,
``elements/_lens_thin.py`` and all of ``propagators/``.  v5.32 (this
change) wires those 41, so the handoff list is now **empty** and the
fail-closed pin in item 4 covers the whole package.  ``_WIRED_SITES``
below is the complete 68-row inventory, and
``test_inventory_is_complete_and_exact`` asserts it matches what an AST
walk of ``lumenairy/`` actually finds -- in **both** directions, so a
future site can neither land unwired nor land wired-but-undeclared.

The user-visible message text changed at only 3 of the 68 sites --
``compute_psf`` ('pupil'), ``compute_otf`` / ``compute_mtf`` ('psf').
The other 65 declare ``'field'``, which was the pre-fix default, so
their text is byte-identical and the win is that the intent is now
machine-checkable instead of a prose comment.

**One recorded intent was wrong and is corrected here.**  The ce0a265
handoff table recorded ``shack_hartmann`` as ``'pupil'``, taken from
that function's own in-source comment ("Input kind: 'pupil' (the SH-WFS
measures a complex pupil-plane field)").  It is wired ``'field'``
instead -- see ``test_shack_hartmann_declares_field_not_pupil`` for the
reasoning and the measurement.

Author: Andrew Traverso -- v5.31 / audit A-9 (wave W4);
completed v5.32 / audit W5-1 (wave W5)
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from lumenairy._validation import _INPUT_KINDS, _check_2d_scalar_field

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# The wired inventory -- ALL 68 call sites as of v5.32.  Built by an AST
# walk of ``lumenairy/`` (see the audit record); each row is
# (module_relpath, guarded_arg, fn_name_literal, expected_input_kind).
#
# The (module, arg, fn_name) triple is unique across all 68 rows, which is
# what lets ``test_wired_site_declares_expected_input_kind`` demand
# exactly one AST match per row.
#
# Kept in package-walk order (sorted by module path, then source order
# within a module) so a diff against a fresh AST dump is readable.
# ---------------------------------------------------------------------------

_WIRED_SITES = (
    ('lumenairy/analysis/ao.py', 'E_in', 'apply_dm', 'field'),
    ('lumenairy/analysis/ao.py', 'E_in', 'DeformableMirror.apply', 'field'),
    ('lumenairy/analysis/beam_stats.py', 'E', 'beam_d4sigma', 'field'),
    ('lumenairy/analysis/beam_stats.py', 'E', 'M2', 'field'),
    ('lumenairy/analysis/coherence.py', 'object_field', 'koehler_image',
     'field'),
    ('lumenairy/analysis/coherence.py', 'object_field',
     'extended_source_image', 'field'),
    # v5.32: the two former handoff sites in analysis/detector.py.
    # ``shack_hartmann`` is 'field', NOT the recorded 'pupil' -- see
    # ``test_shack_hartmann_declares_field_not_pupil``.
    ('lumenairy/analysis/detector.py', 'E', 'apply_detector', 'field'),
    ('lumenairy/analysis/detector.py', 'E', 'shack_hartmann', 'field'),
    ('lumenairy/analysis/opd.py', 'E', 'wave_opd_2d', 'field'),
    # The three sites whose user-visible wording actually changes:
    ('lumenairy/analysis/psf_mtf_otf.py', 'pupil', 'compute_psf', 'pupil'),
    ('lumenairy/analysis/psf_mtf_otf.py', 'psf', 'compute_otf', 'psf'),
    ('lumenairy/analysis/psf_mtf_otf.py', 'psf', 'compute_mtf', 'psf'),
    ('lumenairy/analysis/psf_mtf_otf.py', 'E', 'encircled_energy_curve',
     'field'),
    ('lumenairy/analysis/psf_mtf_otf.py', 'E', 'encircled_energy_radius',
     'field'),
    ('lumenairy/analysis/strehl.py', 'E', 'strehl_ratio', 'field'),
    ('lumenairy/analysis/strehl.py', 'E_ref', 'strehl_ratio', 'field'),
    ('lumenairy/analysis/strehl.py', 'E', 'coupling_efficiency', 'field'),
    ('lumenairy/analysis/strehl.py', 'mode', 'coupling_efficiency', 'field'),
    ('lumenairy/elements/_lens_real.py', 'E_in', 'apply_real_lens', 'field'),
    # v5.32: the six former handoff sites in elements/_lens_thin.py.
    ('lumenairy/elements/_lens_thin.py', 'E_in', 'apply_thin_lens', 'field'),
    ('lumenairy/elements/_lens_thin.py', 'E_in', 'apply_spherical_lens',
     'field'),
    ('lumenairy/elements/_lens_thin.py', 'E_in', 'apply_aspheric_lens',
     'field'),
    ('lumenairy/elements/_lens_thin.py', 'E_in', 'apply_cylindrical_lens',
     'field'),
    ('lumenairy/elements/_lens_thin.py', 'E_in', 'apply_grin_lens', 'field'),
    ('lumenairy/elements/_lens_thin.py', 'E_in', 'apply_axicon', 'field'),
    ('lumenairy/elements/_lens_traced.py', 'E_in', 'apply_real_lens_traced',
     'field'),
    ('lumenairy/elements/_lens_traced.py', 'E_in',
     'apply_real_lens_traced_segmented', 'field'),
    ('lumenairy/elements/_lens_traced_multibranch.py', 'E_in',
     'apply_real_lens_traced_multibranch', 'field'),
    ('lumenairy/elements/_lens_traced_uniform.py', 'E_in',
     'apply_real_lens_traced_uniform', 'field'),
    ('lumenairy/elements/lenses_gbd.py', 'E_in', 'apply_real_lens_gbd',
     'field'),
    ('lumenairy/elements/lenses_maslov.py', 'E_in', 'apply_real_lens_maslov',
     'field'),
    ('lumenairy/optimize/jax_merits.py', 'E_in', 'optimize_traced_geometry',
     'field'),
    # v5.32: the 33 former handoff sites in propagators/ (the module's own
    # 2 vector-diffraction 'pupil' sites were wired back in v4.15.5 and are
    # listed at the end of this block).
    ('lumenairy/propagators/asm.py', 'E_in', 'angular_spectrum_propagate',
     'field'),
    ('lumenairy/propagators/asm.py', 'E_in',
     'angular_spectrum_propagate_tilted', 'field'),
    ('lumenairy/propagators/carrier.py', 'E_env',
     'propagate_carrier_referenced', 'field'),
    ('lumenairy/propagators/carrier.py', 'E_env',
     'carrier_referenced_reconstruct', 'field'),
    ('lumenairy/propagators/carrier.py', 'E_full',
     'carrier_referenced_envelope', 'field'),
    ('lumenairy/propagators/carrier.py', 'E_full',
     'carrier_referenced_fit_radius', 'field'),
    ('lumenairy/propagators/carrier.py', 'E_env',
     'carrier_referenced_aperture', 'field'),
    ('lumenairy/propagators/carrier.py', 'env',
     'carrier_referenced_focus_readout', 'field'),
    ('lumenairy/propagators/carrier.py', 'E_full',
     'carrier_referenced_exact_focus_readout', 'field'),
    ('lumenairy/propagators/carrier.py', 'E_in',
     'propagate_traced_carrier_chain', 'field'),
    ('lumenairy/propagators/dispatch.py', 'E_in', 'propagate', 'field'),
    ('lumenairy/propagators/fga.py', 'E_in', 'apply_real_lens_fga', 'field'),
    ('lumenairy/propagators/fga.py', 'E_in', 'apply_real_lens_auto', 'field'),
    ('lumenairy/propagators/fga.py', 'E_in', 'apply_real_lens_universal',
     'field'),
    ('lumenairy/propagators/fresnel.py', 'E_in', 'fresnel_tf_propagate',
     'field'),
    ('lumenairy/propagators/fresnel.py', 'E_in', 'fresnel_propagate',
     'field'),
    ('lumenairy/propagators/fresnel.py', 'E_in', 'fraunhofer_propagate',
     'field'),
    ('lumenairy/propagators/gbd.py', 'E_in', 'recommend_gbd_sampling',
     'field'),
    ('lumenairy/propagators/gbd.py', 'E_in', 'converge_gbd_sampling',
     'field'),
    ('lumenairy/propagators/gbd.py', 'E_in', 'decompose_field_adaptive',
     'field'),
    ('lumenairy/propagators/gbd.py', 'E', 'gbd_field_to_asm', 'field'),
    ('lumenairy/propagators/gbd.py', 'E', 'asm_field_to_gbd', 'field'),
    ('lumenairy/propagators/gbd.py', 'E', 'match_global_phase', 'field'),
    ('lumenairy/propagators/gbd.py', 'E_in',
     'propagate_gbd_freespace_spectral', 'field'),
    ('lumenairy/propagators/gbd.py', 'E_in', 'propagate_gbd_freespace_csp',
     'field'),
    ('lumenairy/propagators/mft.py', 'E_in', 'angular_spectrum_propagate_mft',
     'field'),
    ('lumenairy/propagators/mft.py', 'E_in', 'resample_field', 'field'),
    ('lumenairy/propagators/mft.py', 'E_in', 'fresnel_propagate_mft',
     'field'),
    ('lumenairy/propagators/mft.py', 'E_in', 'fraunhofer_propagate_mft',
     'field'),
    ('lumenairy/propagators/rs.py', 'E_in', 'rayleigh_sommerfeld_propagate',
     'field'),
    ('lumenairy/propagators/sas.py', 'E_in',
     'scalable_angular_spectrum_propagate', 'field'),
    ('lumenairy/propagators/system.py', 'E_in', 'propagate_through_system',
     'field'),
    ('lumenairy/propagators/system.py', 'E_in',
     'propagate_through_system_jax', 'field'),
    # The original v4.15.5 pair -- the only 'pupil' sites outside
    # ``compute_psf``, and the reason ``input_kind`` exists at all.
    ('lumenairy/propagators/vector_diffraction.py', 'pupil',
     'richards_wolf_focus', 'pupil'),
    ('lumenairy/propagators/vector_diffraction.py', 'pupil',
     'debye_wolf_psf', 'pupil'),
    ('lumenairy/raytrace/from_field.py', 'E', 'rays_from_field', 'field'),
)

# v5.32: **EMPTY**.  ce0a265 parked 41 call sites here while sibling
# agents owned ``analysis/detector.py``, ``elements/_lens_thin.py`` and
# ``propagators/``; every one of those is now wired and listed in
# ``_WIRED_SITES`` above, so the structural fail-closed pin
# (``test_no_in_scope_guard_call_omits_input_kind``) applies to the whole
# of ``lumenairy/`` with no exemptions.
#
# ``str.startswith(())`` is False for every string, so an empty tuple
# exempts nothing -- the pin below is genuinely package-wide.
#
# DO NOT re-add entries to make a failing pin pass.  A new unwired call
# site is the A-9 defect recurring; wire it and add its row above.  (The
# three former "vacuous" members -- ``io/storage.py``,
# ``raytrace/seidel.py``, ``elements/rcwa/`` -- were listed by ce0a265
# only to save the reader re-deriving that they hold zero call sites;
# they need no exemption and are dropped.)
_HANDOFF_PREFIXES: tuple[str, ...] = ()

_BAD_3D = np.ones((3, 8, 8), dtype=np.complex128)


def _site_id(row) -> str:
    rel, arg, fn_name, kind = row
    return f"{Path(rel).name}::{fn_name}({arg})->{kind}"


_SITE_PARAMS = [pytest.param(r, id=_site_id(r)) for r in _WIRED_SITES]


@lru_cache(maxsize=None)
def _guard_calls(rel: str):
    """AST-extract every ``_check_2d_scalar_field`` call in ``rel``.

    Returns a tuple of ``(guarded_arg_src, fn_name_literal,
    input_kind_literal_or_None, lineno)``.
    """
    path = _REPO_ROOT / rel
    src = path.read_text(encoding='utf-8')
    tree = ast.parse(src, filename=str(path))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, 'id', None) or getattr(fn, 'attr', None)
        if name != '_check_2d_scalar_field':
            continue
        arg_src = ast.unparse(node.args[0]) if node.args else '?'
        fn_lit = (node.args[1].value
                  if len(node.args) > 1
                  and isinstance(node.args[1], ast.Constant)
                  else None)
        kind = None
        for kw in node.keywords:
            if kw.arg == 'input_kind' and isinstance(kw.value, ast.Constant):
                kind = kw.value.value
        out.append((arg_src, fn_lit, kind, node.lineno))
    return tuple(out)


def _iter_package_files():
    for path in sorted((_REPO_ROOT / 'lumenairy').rglob('*.py')):
        yield path.relative_to(_REPO_ROOT).as_posix(), path


# ---------------------------------------------------------------------------
# A-9.1 -- the vocabulary is closed and checked, not prose-only
# ---------------------------------------------------------------------------

def test_input_kinds_vocabulary_is_a_closed_frozenset():
    """``_INPUT_KINDS`` must be an immutable set of plain strings that
    covers the three kinds the library actually guards.

    Pinned as a *superset* check, not equality: a sibling agent wiring
    an intensity-domain entry point may legitimately need to add a
    value, and that should not require editing this pin.
    """
    assert isinstance(_INPUT_KINDS, frozenset), (
        f"_INPUT_KINDS must be a frozenset (immutable, cheap membership "
        f"test on the propagator hot path); got {type(_INPUT_KINDS)}.")
    assert all(isinstance(v, str) for v in _INPUT_KINDS)
    assert {'field', 'psf', 'pupil'} <= _INPUT_KINDS, (
        f"_INPUT_KINDS must cover the three wired kinds; got "
        f"{sorted(_INPUT_KINDS)}.")


def test_default_input_kind_is_field_backcompat():
    """The default stays ``'field'``.

    As of v5.32 no call site in ``lumenairy/`` relies on it -- all 68
    declare explicitly -- but the default is still load-bearing for two
    reasons: the v4.15.5 back-compat pin
    (``test_v4_15_5_agent_b.py::
    test_validation_helper_input_kind_default_field_backcompat``) asserts
    the historic wording, and ``_check_2d_scalar_field`` is a private
    helper that downstream forks / in-flight branches call positionally.
    Removing the default would be an unrelated API break, so the
    fail-closed pin below -- not the signature -- is what keeps sites
    from silently defaulting.
    """
    with pytest.raises(ValueError) as ei:
        _check_2d_scalar_field(_BAD_3D, 'fake_fn')
    assert 'expected 2-D complex field of shape' in str(ei.value)


# ---------------------------------------------------------------------------
# A-9.2 -- junk ``input_kind`` raises ValueError naming value + allowed set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('junk', [
    'feild',        # typo
    'pupils',       # plural
    'intensity',    # borrowed from a sibling knob's vocabulary
    'Field',        # wrong case
    'FIELD',
    '',             # empty
    'field ',       # trailing space
    None,           # wrong type
    42,
])
def test_junk_input_kind_raises_valueerror_naming_value_and_allowed(junk):
    """House rule: an enum-valued knob rejects out-of-vocabulary input
    with a ``ValueError`` that prints the offending value AND the
    allowed set.

    Pre-fix, ``input_kind`` was interpolated straight into the message,
    so ``input_kind='feild'`` produced a *silently* misleading
    "expected 2-D complex feild of shape (Ny, Nx)" -- the guard fired,
    but told the user to pass something that does not exist.
    """
    with pytest.raises(ValueError) as ei:
        _check_2d_scalar_field(_BAD_3D, 'fake_fn', input_kind=junk)
    msg = str(ei.value)
    assert msg.startswith('fake_fn: '), (
        f"message must carry the CONVENTIONS.md ``fn_name: `` prefix; "
        f"got {msg!r}")
    assert repr(junk) in msg, (
        f"message must name the offending value {junk!r}; got {msg!r}")
    for allowed in _INPUT_KINDS:
        assert repr(allowed) in msg, (
            f"message must name the allowed value {allowed!r}; got "
            f"{msg!r}")


def test_junk_input_kind_checked_before_the_input_itself():
    """A junk ``input_kind`` is a *library* bug (the string comes from
    lumenairy's own call sites, never from the end user), so it is
    reported even when ``E`` is a perfectly good 2-D field -- rather
    than being masked until someone happens to pass a bad array.
    """
    good = np.ones((8, 8), dtype=np.complex128)
    with pytest.raises(ValueError, match='input_kind must be one of'):
        _check_2d_scalar_field(good, 'fake_fn', input_kind='feild')


def test_junk_input_kind_beats_the_mcf_rejection():
    """Ordering pin: the contract check runs before the MCF
    ``TypeError``, so a call site with a broken ``input_kind`` cannot
    hide behind a user's partial-coherence input.
    """
    from lumenairy.sources.core import PartialCoherenceMCF

    mcf = PartialCoherenceMCF.__new__(PartialCoherenceMCF)
    with pytest.raises(ValueError, match='input_kind must be one of'):
        _check_2d_scalar_field(mcf, 'fake_fn', input_kind='feild')


@pytest.mark.parametrize('site', _SITE_PARAMS)
def test_wired_site_junk_input_kind_raises(site):
    """Per-wired-site junk pin: the guard behind every newly-wired entry
    point rejects an out-of-vocabulary kind under that site's own
    ``fn_name``, so the error is attributable to the offending call
    site rather than to the shared helper.
    """
    _rel, _arg, fn_name, _kind = site
    with pytest.raises(ValueError) as ei:
        _check_2d_scalar_field(_BAD_3D, fn_name, input_kind='feild')
    msg = str(ei.value)
    assert msg.startswith(f"{fn_name}: ")
    assert "'feild'" in msg
    assert sorted(_INPUT_KINDS)[0] in msg


# ---------------------------------------------------------------------------
# A-9.3 -- every in-scope site declares the right kind explicitly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('site', _SITE_PARAMS)
def test_wired_site_declares_expected_input_kind(site):
    """AST pin: the call site passes an explicit ``input_kind`` literal
    matching the semantics of the guarded argument.

    Relying on the default is what let ``compute_psf`` (a pupil) and
    ``compute_otf`` / ``compute_mtf`` (a PSF intensity) report "field"
    for five minor releases.
    """
    rel, arg, fn_name, kind = site
    matches = [c for c in _guard_calls(rel)
               if c[1] == fn_name and c[0] == arg]
    assert len(matches) == 1, (
        f"{rel}: expected exactly 1 ``_check_2d_scalar_field({arg}, "
        f"{fn_name!r}, ...)`` call; found {len(matches)}.  If the site "
        f"moved or the argument was renamed, update _WIRED_SITES.")
    arg_src, _fn, declared, lineno = matches[0]
    assert declared is not None, (
        f"{rel}:{lineno} {fn_name}: ``_check_2d_scalar_field({arg_src}, "
        f"...)`` does not pass ``input_kind``.  Pass "
        f"``input_kind={kind!r}`` explicitly -- the default is 'field' "
        f"and a silent default is exactly the A-9 defect.")
    assert declared == kind, (
        f"{rel}:{lineno} {fn_name}: guards {arg_src!r} with "
        f"``input_kind={declared!r}``; expected {kind!r}.")
    assert declared in _INPUT_KINDS


@pytest.mark.parametrize('site', _SITE_PARAMS)
def test_wired_site_message_names_declared_kind(site):
    """The declared kind reaches the user-visible message: rejecting a
    3-D input at that site reads "expected 2-D complex <kind>".
    """
    _rel, _arg, fn_name, kind = site
    with pytest.raises(ValueError) as ei:
        _check_2d_scalar_field(_BAD_3D, fn_name, input_kind=kind)
    msg = str(ei.value)
    assert f"expected 2-D complex {kind} of shape" in msg, (
        f"{fn_name}: message must name the declared kind {kind!r}; got "
        f"{msg!r}")


def test_no_in_scope_guard_call_omits_input_kind():
    """Fail-closed structural pin (the A-9 counter-measure).

    **Every** ``_check_2d_scalar_field`` call in ``lumenairy/`` must pass
    ``input_kind``.  As of v5.32 ``_HANDOFF_PREFIXES`` is empty, so this
    is package-wide with no exemptions (it is retained as a mechanism
    only because a future genuinely-scoped handoff may need it, and
    ``test_handoff_list_is_empty`` guards against casual use).

    A new entry point that copies the old
    ``_check_2d_scalar_field(E, 'my_fn')`` form trips this pin instead of
    silently re-opening the 2-of-68 gap.
    """
    unwired = []
    for rel, _path in _iter_package_files():
        if rel.startswith(_HANDOFF_PREFIXES):
            continue
        for arg_src, fn_lit, kind, lineno in _guard_calls(rel):
            if kind is None:
                unwired.append(
                    f"{rel}:{lineno} _check_2d_scalar_field({arg_src}, "
                    f"{fn_lit!r}) -- missing input_kind=")
    assert not unwired, (
        "Guard call sites missing an explicit ``input_kind`` (audit "
        "A-9).  Pass one of "
        f"{sorted(_INPUT_KINDS)}:\n  - " + "\n  - ".join(unwired))


def test_handoff_list_is_empty():
    """The A-9 rollout is complete: nothing is exempt from the
    fail-closed pin any more.

    ce0a265 parked 41 sites in ``_HANDOFF_PREFIXES`` because sibling
    agents owned those files.  v5.32 wired them.  An entry reappearing
    here would silently re-open the very hole item 4 exists to close, so
    the emptiness is itself pinned -- re-adding a prefix to quiet a
    failing pin now has to defeat this assertion first, which forces the
    conversation.
    """
    assert _HANDOFF_PREFIXES == (), (
        f"_HANDOFF_PREFIXES must stay empty; got {_HANDOFF_PREFIXES}.  A "
        f"new unwired call site is the A-9 defect recurring -- wire it "
        f"and add its row to _WIRED_SITES rather than exempting its "
        f"module.")


def test_inventory_is_complete_and_exact():
    """Completeness pin (the counter-measure that makes 68/68 assertable).

    An AST walk of ``lumenairy/`` must find **exactly** the sites listed
    in ``_WIRED_SITES`` -- checked in BOTH directions:

    * a guard call the inventory does not list => a new entry point
      landed without being declared here (even if it happens to pass
      ``input_kind``, nobody reviewed *which* kind it should be);
    * an inventory row with no matching guard call => a site was removed
      or renamed and the pin has gone stale, silently reducing coverage.

    Either direction alone is escapable, which is how "2 of 68" survived
    five minor releases.  Together they mean the count in this file's
    docstring cannot drift from reality.
    """
    found = set()
    for rel, _path in _iter_package_files():
        for arg_src, fn_lit, _kind, _lineno in _guard_calls(rel):
            found.add((rel, arg_src, fn_lit))
    listed = {(rel, arg, fn) for rel, arg, fn, _k in _WIRED_SITES}

    undeclared = sorted(found - listed)
    stale = sorted(listed - found)
    assert not undeclared, (
        "guard call sites present in lumenairy/ but MISSING from "
        "_WIRED_SITES -- add a row (module, arg, fn_name, input_kind) "
        "for each, having decided which kind it guards:\n  - "
        + "\n  - ".join(f"{r}: {f}({a})" for r, a, f in undeclared))
    assert not stale, (
        "_WIRED_SITES rows with no matching guard call -- the site moved, "
        "was renamed, or was deleted; update the inventory:\n  - "
        + "\n  - ".join(f"{r}: {f}({a})" for r, a, f in stale))
    assert found == listed


def test_all_sixty_eight_sites_are_wired():
    """The headline counter: 68 call sites, 68 explicit declarations, 0
    relying on the default.

    The literal 68 is asserted (not merely derived) so that *reducing*
    the inventory cannot make the rollout look complete -- shrinking
    ``_WIRED_SITES`` to match a regression would trip this, and growing
    the package legitimately requires a deliberate bump here plus a row
    above.
    """
    total = sum(len(_guard_calls(rel)) for rel, _p in _iter_package_files())
    declared = sum(
        1 for rel, _p in _iter_package_files()
        for c in _guard_calls(rel) if c[2] is not None)
    assert len(_WIRED_SITES) == 68, (
        f"_WIRED_SITES must hold all 68 sites; got {len(_WIRED_SITES)}.")
    assert total == 68, (
        f"expected 68 ``_check_2d_scalar_field`` calls in lumenairy/; "
        f"found {total}.  If an entry point was legitimately added or "
        f"removed, wire it, update _WIRED_SITES, and bump this count in "
        f"the same change (and in this file's docstring).")
    assert declared == total, (
        f"only {declared} of {total} guard calls declare input_kind.")


def test_declared_kind_distribution():
    """Diagnostic counter-pin: 63 'field' / 3 'pupil' / 2 'psf'.

    Documents *which* sites are the interesting ones, and catches a
    careless bulk edit that flips a kind -- e.g. a future sweep
    "normalising" ``compute_psf`` back to 'field' would keep every other
    pin in this file green while undoing the headline A-9 fix.

    The 3 'pupil' sites are exactly the ones whose parameter IS named
    ``pupil`` (``compute_psf``, ``richards_wolf_focus``,
    ``debye_wolf_psf``); the 2 'psf' sites are ``compute_otf`` /
    ``compute_mtf``.  Everything else guards a complex field.
    """
    from collections import Counter

    counts = Counter(kind for _r, _a, _f, kind in _WIRED_SITES)
    assert dict(counts) == {'field': 63, 'pupil': 3, 'psf': 2}, (
        f"declared-kind distribution drifted: {dict(counts)}")

    pupil_fns = {fn for _r, _a, fn, k in _WIRED_SITES if k == 'pupil'}
    assert pupil_fns == {'compute_psf', 'richards_wolf_focus',
                         'debye_wolf_psf'}, pupil_fns
    psf_fns = {fn for _r, _a, fn, k in _WIRED_SITES if k == 'psf'}
    assert psf_fns == {'compute_otf', 'compute_mtf'}, psf_fns

    # Every 'pupil' site guards an argument literally named ``pupil``,
    # and every 'psf' site one named ``psf``.  That correspondence is the
    # rule the shack_hartmann correction below rests on.
    for _r, arg, fn, kind in _WIRED_SITES:
        if kind in ('pupil', 'psf'):
            assert arg == kind, (
                f"{fn}: declares input_kind={kind!r} but guards an "
                f"argument named {arg!r}; the two must agree (see "
                f"test_shack_hartmann_declares_field_not_pupil).")


def test_shack_hartmann_declares_field_not_pupil():
    """Recorded-intent CORRECTION (v5.32).

    The ce0a265 handoff table recorded ``analysis/detector.py``'s
    ``shack_hartmann`` as ``'pupil'``, sourced from that function's own
    comment: "Input kind: 'pupil' (the SH-WFS measures a complex
    pupil-plane field)".  Verified against the actual contract, it is
    ``'field'``:

    1. The gloss argues itself out of its own conclusion -- "pupil-plane"
       names the PLANE; the thing the argument carries is a FIELD.  The
       Parameters entry agrees verbatim: "E : ndarray, complex, shape
       (N, N) / Input field at the lenslet array plane."
    2. ``input_kind`` picks the noun in the rejection message, so it must
       match what the caller passed.  Declaring 'pupil' would tell
       someone who passed ``E`` that a "2-D complex pupil" was expected
       -- re-creating in mirror image the exact A-9 defect
       (``compute_psf(pupil, ...)`` reporting "field") this rollout
       closes.
    3. Library-wide correspondence (pinned in
       ``test_declared_kind_distribution``): only the sites whose
       parameter IS named ``pupil`` declare 'pupil'.
    4. Physically it must be a metrically-scaled field, not a
       dimensionless pupil function: the entry point takes ``dx`` and
       ``wavelength`` and propagates each sub-aperture with the
       bandlimited angular-spectrum kernel at ``z = lenslet_focal``.

    Consequence: ``shack_hartmann``'s rejection wording is UNCHANGED from
    pre-A-9 ('field' was the default), so this correction also keeps the
    site off the "message text changed" list -- 3 sites, not 4.
    """
    from lumenairy.analysis import shack_hartmann

    matches = [c for c in _guard_calls('lumenairy/analysis/detector.py')
               if c[1] == 'shack_hartmann']
    assert len(matches) == 1
    assert matches[0][2] == 'field', (
        f"shack_hartmann must declare input_kind='field' (see this "
        f"test's docstring); got {matches[0][2]!r}.")

    with pytest.raises(ValueError) as ei:
        shack_hartmann(_BAD_3D, 5e-6, 633e-9, 40e-6, 5e-3)
    msg = str(ei.value)
    assert 'expected 2-D complex field of shape' in msg, msg
    assert 'complex pupil' not in msg, msg

    # The superseded claim must not survive as an unqualified assertion
    # in the source: the next reader who greps that comment must find the
    # correction attached to it, not just a comment silently contradicted
    # by the line beneath it.
    src = (_REPO_ROOT / 'lumenairy/analysis/detector.py').read_text(
        encoding='utf-8')
    assert 'superseding' in src, (
        "detector.py must record that the v4.15.5 \"Input kind: 'pupil'\" "
        "comment was superseded, not merely contradict it.")


def test_stale_input_kind_todo_is_gone():
    """The ``compute_psf`` marker comment ("would be ideal once Agent
    B's parameterised ``_check_2d_scalar_field`` lands") must not
    survive the rollout that satisfies it.
    """
    src = (_REPO_ROOT / 'lumenairy/analysis/psf_mtf_otf.py').read_text(
        encoding='utf-8')
    assert 'would be ideal once' not in src, (
        "psf_mtf_otf.py still carries the stale 'would be ideal once "
        "Agent B's parameterised helper lands' marker; the helper "
        "landed in v4.15.5 and is wired as of v5.31.")
    assert 'correct in the interim' not in src


# ---------------------------------------------------------------------------
# A-9.4 -- bit-identity spot checks (wiring is message-only)
# ---------------------------------------------------------------------------

_N = 32
_DX = 5e-6
_WL = 633e-9
_x = (np.arange(_N) - _N / 2) * _DX
_X, _Y = np.meshgrid(_x, _x)
_E = np.exp(-(_X ** 2 + _Y ** 2) / (40e-6) ** 2).astype(np.complex128)
_E_AB = _E * np.exp(1j * 0.7 * (_X / (_N * _DX / 2)) ** 2)


@pytest.mark.parametrize('kind', sorted(_INPUT_KINDS))
def test_valid_input_accepted_for_every_kind(kind):
    """``input_kind`` selects message wording only -- it never changes
    what the guard accepts.
    """
    assert _check_2d_scalar_field(_E, 'fake_fn', input_kind=kind) is None


def test_compute_psf_none_normalisation_is_bit_identical_to_formula():
    """``compute_psf(normalize='none')`` is exactly
    ``|fftshift(fft2(ifftshift(pupil)))|**2``.  Wiring
    ``input_kind='pupil'`` must not perturb a single bit of it.
    """
    from lumenairy.analysis import compute_psf

    psf, dx_psf = compute_psf(_E_AB, _WL, 25e-3, _DX, normalize='none')
    ref = np.abs(np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(_E_AB)))) ** 2
    assert np.array_equal(psf, ref)
    assert dx_psf == _WL * 25e-3 / (_N * _DX)


def test_compute_mtf_is_bit_identical_to_abs_otf():
    """``compute_mtf`` is defined as ``|OTF|``; both are now wired with
    ``input_kind='psf'`` and the identity is exact.
    """
    from lumenairy.analysis import compute_mtf, compute_otf, compute_psf

    psf, _ = compute_psf(_E_AB, _WL, 25e-3, _DX)
    assert np.array_equal(compute_mtf(psf), np.abs(compute_otf(psf)))


def test_compute_otf_dc_is_exactly_one():
    """OTF normalisation pin (unchanged by the wiring): the DC bin is
    exactly 1+0j after the divide.
    """
    from lumenairy.analysis import compute_otf, compute_psf

    psf, _ = compute_psf(_E_AB, _WL, 25e-3, _DX)
    otf = compute_otf(psf)
    dc = otf[otf.shape[0] // 2, otf.shape[1] // 2]
    assert complex(dc) == 1 + 0j


def test_apply_dm_is_bit_identical_to_method_and_formula():
    """``apply_dm`` (wired) delegates to ``DeformableMirror.apply``
    (also wired); both remain exactly ``E * exp(1j * scale * phase)``.
    """
    from lumenairy.analysis import DeformableMirror, apply_dm

    dm = DeformableMirror(n_actuators=3, pitch=8 * _DX, N=_N, dx=_DX)
    dm.command = np.linspace(-0.3, 0.3, dm.command.size).reshape(
        dm.command.shape)
    out = apply_dm(_E, dm, scale=0.5)
    assert np.array_equal(out, dm.apply(_E, scale=0.5))
    assert np.array_equal(out, _E * np.exp(1j * 0.5 * dm.phase()))


def test_strehl_and_coupling_self_reference_unchanged():
    """A field against itself: Strehl is exactly 1.0 and coupling is
    1.0 to round-off.  Both entry points guard TWO arguments and both
    guards are now wired.
    """
    from lumenairy.analysis import coupling_efficiency, strehl_ratio

    assert strehl_ratio(_E_AB, _E_AB, _DX) == 1.0
    assert coupling_efficiency(_E_AB, _E_AB, _DX) == pytest.approx(
        1.0, abs=1e-12)


def test_encircled_energy_radius_inverts_the_curve_exactly():
    """The radius is the exact inverse of the curve (v5.30, audit A-2);
    both are wired with ``input_kind='field'`` and the shared
    ``_ee_sorted_cumulative`` path is untouched.
    """
    from lumenairy.analysis import (
        encircled_energy_curve,
        encircled_energy_radius,
    )

    r, ee = encircled_energy_curve(_E_AB, _DX, n_radii=9)
    assert np.all(np.diff(ee) >= -1e-12)
    r865 = encircled_energy_radius(_E_AB, _DX, threshold=0.865)
    assert 0.0 < r865 <= float(r[-1])


# ---------------------------------------------------------------------------
# A-9.5 -- the three message changes, through the public API
# ---------------------------------------------------------------------------

def test_compute_psf_rejection_says_pupil_not_field():
    """The headline A-9 fix: ``compute_psf`` consumes a pupil, so a
    wrong-rank input must not be described as a "field".
    """
    from lumenairy.analysis import compute_psf

    with pytest.raises(ValueError) as ei:
        compute_psf(_BAD_3D, _WL, 25e-3, _DX)
    msg = str(ei.value)
    assert 'expected 2-D complex pupil of shape' in msg, msg
    assert 'complex field' not in msg, msg


@pytest.mark.parametrize('fn_name', ['compute_otf', 'compute_mtf'])
def test_otf_mtf_rejection_says_psf_not_field(fn_name):
    """``compute_otf`` / ``compute_mtf`` consume an intensity PSF."""
    import lumenairy.analysis as an

    with pytest.raises(ValueError) as ei:
        getattr(an, fn_name)(_BAD_3D)
    msg = str(ei.value)
    assert 'expected 2-D complex psf of shape' in msg, msg
    assert 'complex field' not in msg, msg

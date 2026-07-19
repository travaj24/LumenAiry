"""G10 / S5-9 (AUDIT_V5_24_2) -- shared LayerSpec is a tracked gap.

S5-9 is a LARGE cross-engine feature-gap (no neutral geometry object
that RCWA / PMM / Berreman / EME / BOR stacks can all consume).  Per the
audit-remediation scope rules it is DEFERRED: documented + tracked
rather than half-built in a hygiene pass.

This regression test pins the tracking so the deferral cannot silently
evaporate: the gap must remain visible both in ``ROADMAP.md`` and in the
``lumenairy.elements`` module docstring, keyed by its finding id.  It
fails before the doc + tracking edits and passes after.
"""
from __future__ import annotations

from pathlib import Path

import lumenairy.elements as elements

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ROADMAP = _REPO_ROOT / 'ROADMAP.md'


def test_roadmap_tracks_s5_9_layerspec_gap():
    text = _ROADMAP.read_text(encoding='utf-8')
    assert 'S5-9' in text, 'ROADMAP.md no longer tracks S5-9.'
    assert 'LayerSpec' in text, (
        'ROADMAP.md S5-9 entry lost the LayerSpec keyword.')


def test_elements_docstring_tracks_s5_9():
    doc = elements.__doc__ or ''
    assert 'S5-9' in doc, (
        'lumenairy.elements docstring no longer documents the S5-9 '
        'shared-LayerSpec gap.')
    assert 'LayerSpec' in doc

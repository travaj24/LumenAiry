"""Splice this verification's eight node ids into ``.test_durations``.

pytest-split's own ``--store-durations`` REPLACES the file with just the run's
own tests; this merges instead, and re-sorts, so the diff is an insertion.
Durations are the WIN readings under load (2026-09-11); the WSL readings are
1.1x-1.9x of them and pytest-split only needs the relative weights.
"""
from __future__ import annotations

import json
import os

NEW = {
    "tests/unit/test_verify_slant_anchor_v1_v2_o2.py::"
    "test_verify_v1_no_eighth_route_carries_a_shear_into_the_jnp_twin": 7.24,
    "tests/unit/test_verify_slant_anchor_v1_v2_o2.py::"
    "test_verify_o2_the_two_populations_separate_on_an_independent_family":
        6.10,
    "tests/unit/test_verify_slant_anchor_v1_v2_o2.py::"
    "test_verify_o2_the_refusal_is_the_same_with_the_census_armed_or_not":
        3.19,
    "tests/unit/test_verify_slant_anchor_v1_v2_o2.py::"
    "test_verify_v2_a_second_sheared_layer_sits_at_the_ACCUMULATED_walk":
        2.77,
    "tests/unit/test_verify_slant_anchor_v1_v2_o2.py::"
    "test_verify_v2_the_anchor_stays_unimodular_through_a_wood_anomaly": 0.14,
    "tests/unit/test_verify_slant_anchor_v1_v2_o2.py::"
    "test_verify_v2_the_default_factorization_exposes_no_transmitted_field":
        0.08,
    "tests/unit/test_verify_slant_anchor_v1_v2_o2.py::"
    "test_verify_v2_a_net_zero_walk_is_the_exact_identity_through_a_solve":
        0.05,
    "tests/unit/test_verify_slant_anchor_v1_v2_o2.py::"
    "test_verify_v2_a_sheared_stack_exposes_no_other_frame_bearing_surface":
        0.01,
}

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
P = os.path.join(ROOT, ".test_durations")

with open(P, encoding="utf-8") as fh:
    d = json.load(fh)
before = len(d)
d.update(NEW)
with open(P, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(dict(sorted(d.items())), fh, indent=2)
    fh.write("\n")
print(f"{before} -> {len(d)} ids ({len(NEW)} spliced)")

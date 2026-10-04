"""K1: the readings behind the bars of
``tests/unit/test_jax_symmetric_point_gradients.py`` -- every test function
run here with its ``_rel`` instrumented, so the measured relative errors /
changes it compares against its bars are recorded per build.

    python k1_test_readings.py
"""
import os
import sys

from _h import HERE, dump

sys.path.insert(0, os.path.join(HERE, "..", "..", "tests", "unit"))
import test_jax_symmetric_point_gradients as M  # noqa: E402
from _pytest.monkeypatch import MonkeyPatch  # noqa: E402

_orig_rel = M._rel
seen = []


def _rec(g, fd):
    v = _orig_rel(g, fd)
    seen.append(v)
    return v


M._rel = _rec
CASES = [
    ("gauge_te", lambda mp: M.test_rcwa_jax_gradient_does_not_depend_on_the_basis_inside_a_cluster("te", mp)),
    ("gauge_tm", lambda mp: M.test_rcwa_jax_gradient_does_not_depend_on_the_basis_inside_a_cluster("tm", mp)),
    ("near_random_te", lambda mp: M.test_rcwa_jax_gradient_near_a_symmetric_cell_without_any_symmetry("te")),
    ("near_random_tm", lambda mp: M.test_rcwa_jax_gradient_near_a_symmetric_cell_without_any_symmetry("tm")),
    ("rcwa_jones_2d", lambda mp: M.test_rcwa_jones_2d_symmetry_breaking_gradient_at_a_symmetric_cell()),
    ("rcwa_stack", lambda mp: M.test_rcwa_stack_symmetry_breaking_gradient_at_a_symmetric_cell()),
    ("berreman", lambda mp: M.test_berreman_traced_tensor_gradient_at_an_isotropic_layer()),
    ("pmm_jones_1d", lambda mp: M.test_pmm_jones_1d_angle_gradient_at_normal_incidence()),
    ("pmm_stack_shared", lambda mp: M.test_pmm_stack_angle_gradient_at_normal_incidence("shared")),
    ("pmm_stack_perlayer", lambda mp: M.test_pmm_stack_angle_gradient_at_normal_incidence("per-layer")),
    ("pmm2d_traced", lambda mp: M.test_pmm2d_hybrid_stack_traced_layout_gradient_at_a_symmetric_cell()),
    ("switch_te", lambda mp: M.test_the_switch_off_is_wrong_at_a_symmetric_point_and_inert_elsewhere("te")),
    ("jones_li", lambda mp: M.test_rcwa_jones_2d_li_on_a_traced_tensor_is_li_at_a_symmetric_cell()),
    ("vmap_mix", lambda mp: M.test_a_vmapped_batch_mixing_symmetric_and_generic_points_is_exact()),
    ("switch_tm", lambda mp: M.test_the_switch_off_is_wrong_at_a_symmetric_point_and_inert_elsewhere("tm")),
]
only = sys.argv[1:]
out = {}
for name, fn in CASES:
    if only and name not in only:
        continue
    seen.clear()
    mp = MonkeyPatch()
    try:
        fn(mp)
        status = "pass"
    except AssertionError as e:
        status = f"FAIL {str(e)[:200]}"
    except Exception as e:  # noqa: BLE001
        status = f"ERROR {type(e).__name__}: {str(e)[:200]}"
    finally:
        mp.undo()
    out[name] = {"status": status, "rel_values": list(seen)}
    print(name, status, ["%.2e" % v for v in seen], flush=True)
# the 'li' forward: jitted traced tensor vs NumPy li, and vs NumPy laurent
if not only or "li_fwd" in only:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from lumenairy.elements.rcwa import rcwa_jones_2d
    rec = {}
    for name, cell in (("c4v", M._BASE), ("random", 2.25 + 0.6 * M._RANDOM)):
        def fj(t, cell=cell):
            e = (jnp.asarray(cell) + t).astype(complex)
            e = e[:, :, None, None] * jnp.eye(3, dtype=complex)[None, None]
            return rcwa_jones_2d(M._P, M._P, e, 1.45, 1.0, 0.45, M._WL,
                                 n_orders_x=3, n_orders_y=3,
                                 formulation="li")[1]
        e_np = (np.asarray(cell).astype(complex)[:, :, None, None]
                * np.eye(3)[None, None])
        R_li = np.asarray(rcwa_jones_2d(M._P, M._P, e_np, 1.45, 1.0, 0.45,
                                        M._WL, n_orders_x=3, n_orders_y=3,
                                        formulation="li")[1])
        R_la = np.asarray(rcwa_jones_2d(M._P, M._P, e_np, 1.45, 1.0, 0.45,
                                        M._WL, n_orders_x=3, n_orders_y=3,
                                        formulation="laurent")[1])
        Rj = np.asarray(jax.jit(fj)(0.0))
        rec[name] = {"jit_vs_numpy_li": float(np.max(np.abs(Rj - R_li))),
                     "numpy_li_vs_laurent": float(np.max(np.abs(R_li - R_la)))}
        print("li_fwd", name, rec[name], flush=True)
    out["li_fwd"] = rec
print(dump("k1_test_readings.json", out))

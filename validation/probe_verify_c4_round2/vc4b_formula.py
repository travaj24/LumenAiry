"""VERIFY-WP-C4 ROUND 2, item 1a -- is the work/entry formula the arithmetic
the dense route ACTUALLY does?

Three checks, none of them a reading of the comment:

1.  **The association order.**  ``_auto_selects_direct`` scores a shape with
    ``min(My*Ny*Nx + My*Nx*Mx,  Ny*Nx*Mx + My*Ny*Mx)``.  ``_direct_matrix_2d``
    picks ``(Wy.E).Wx^T`` when ``cost_y_first <= cost_x_first`` and the other
    association otherwise.  The claim under test is that the executed cost IS
    that ``min``.  Checked by re-deriving both branch costs from the SOURCE of
    ``_direct_matrix_2d`` (its two assignments, read as AST) and by OBSERVING
    which association a live call takes, through a counting wrapper around
    ``xp.matmul`` that records the two operand shapes.
2.  **The kernel-entry count.**  ``entries = My*Ny + Mx*Nx`` is claimed to be
    the number of transcendental kernel entries built.  Checked by counting the
    elements ``np.exp`` is actually called on, through a counting wrapper.
3.  **Which of the two products is the one the association picks**, per shape,
    so a shape whose two costs are close is visible rather than assumed.

    PYTHONPATH=<tree> python vc4b_formula.py <tree>
"""
from __future__ import annotations

import ast
import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402

SHAPES = [
    # (Ny, Nx, My, Mx)
    (256, 256, 8, 8), (2048, 2048, 64, 64), (2048, 64, 64, 2),
    (4096, 128, 128, 4), (128, 4096, 4, 128), (256, 64, 4, 1),
    (64, 256, 1, 4), (1024, 128, 32, 4), (128, 1024, 4, 32),
    (512, 32, 16, 1), (32, 512, 1, 16), (2048, 128, 64, 4),
    (128, 2048, 4, 64), (1000, 1000, 25, 25), (2048, 512, 64, 16),
    (512, 2048, 16, 64), (64, 64, 2, 2), (4096, 32, 128, 1),
]


def rule_flops(ny, nx, my, mx):
    return min(my * ny * nx + my * nx * mx, ny * nx * mx + my * ny * mx)


def rule_entries(ny, nx, my, mx):
    return my * ny + mx * nx


def main(tree):
    import numpy as np
    L.anchor(tree)
    from lumenairy.propagators import _bluestein as B

    out = {'build': L.build(), 'python': sys.version.split()[0],
           'numpy': np.__version__, 'rows': [],
           'source_costs': {}, 'source_matches_rule': None}

    # ---- 1) the two cost expressions, read out of the SOURCE -------------
    src = inspect.getsource(B._direct_matrix_2d)
    tree_ast = ast.parse(src.lstrip())
    costs = {}
    for node in ast.walk(tree_ast):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', '') in
                ('cost_y_first', 'cost_x_first')):
            costs[node.targets[0].id] = ast.unparse(node.value)
    out['source_costs'] = costs
    # Evaluate both source expressions symbolically at every shape and compare
    # with the rule's min().
    ok = True
    for (ny, nx, my, mx) in SHAPES:
        env = {'Ny_in': ny, 'Nx_in': nx, 'N_out_y': my, 'N_out_x': mx}
        cy = eval(costs['cost_y_first'], {}, env)                # noqa: S307
        cx = eval(costs['cost_x_first'], {}, env)                # noqa: S307
        if min(cy, cx) != rule_flops(ny, nx, my, mx):
            ok = False
    out['source_matches_rule'] = ok

    # ---- 2/3) observe the live call --------------------------------------
    for (ny, nx, my, mx) in SHAPES:
        E = (np.random.default_rng(7).standard_normal((ny, nx))
             + 0j).astype(np.complex128)
        alpha = 1.0e3 / float(max(ny, nx, my, mx)) ** 2

        exp_elems = [0]
        matmuls = []
        real_exp, real_matmul = np.exp, np.matmul

        def counted_exp(x, *a, **k):
            exp_elems[0] += int(np.asarray(x).size)
            return real_exp(x, *a, **k)

        def counted_matmul(a1, a2, *a, **k):
            matmuls.append((tuple(np.shape(a1)), tuple(np.shape(a2))))
            return real_matmul(a1, a2, *a, **k)

        np.exp, np.matmul = counted_exp, counted_matmul
        try:
            B._direct_matrix_2d(E, alpha, alpha, my, mx, sign=-1, xp=np)
        finally:
            np.exp, np.matmul = real_exp, real_matmul

        # multiply-adds actually issued, from the observed operand shapes
        observed = sum(a[0] * a[1] * b[1] for a, b in matmuls)
        cy = my * ny * nx + my * nx * mx
        cx = ny * nx * mx + my * ny * mx
        out['rows'].append({
            'shape': f"{ny}x{nx}->{my}x{mx}",
            'Ny': ny, 'Nx': nx, 'My': my, 'Mx': mx,
            'cost_y_first': cy, 'cost_x_first': cx,
            'rule_flops': rule_flops(ny, nx, my, mx),
            'observed_flops': observed,
            'flops_match': observed == rule_flops(ny, nx, my, mx),
            'association_taken': ('y_first' if matmuls
                                  and matmuls[0][0] == (my, ny)
                                  else 'x_first'),
            'association_predicted': ('y_first' if cy <= cx else 'x_first'),
            'matmul_operands': matmuls,
            'rule_entries': rule_entries(ny, nx, my, mx),
            'observed_exp_elements': exp_elems[0],
            'entries_match': exp_elems[0] == rule_entries(ny, nx, my, mx),
            'work_per_entry': rule_flops(ny, nx, my, mx)
                              / rule_entries(ny, nx, my, mx),
        })
        r = out['rows'][-1]
        print(f"{r['shape']:>22s} flops obs={observed:>12d} "
              f"rule={r['rule_flops']:>12d} {'OK' if r['flops_match'] else 'MISMATCH'}"
              f" | entries obs={exp_elems[0]:>10d} rule={r['rule_entries']:>10d}"
              f" {'OK' if r['entries_match'] else 'MISMATCH'}"
              f" | assoc {r['association_taken']}/{r['association_predicted']}"
              f" | w/e={r['work_per_entry']:9.3f}", flush=True)

    out['ALL_flops_match'] = all(r['flops_match'] for r in out['rows'])
    out['ALL_entries_match'] = all(r['entries_match'] for r in out['rows'])
    out['ALL_association_match'] = all(
        r['association_taken'] == r['association_predicted']
        for r in out['rows'])
    print("source expressions == rule min():", out['source_matches_rule'])
    print("ALL flops match      :", out['ALL_flops_match'])
    print("ALL entries match    :", out['ALL_entries_match'])
    print("ALL association match:", out['ALL_association_match'])
    L.write(out, os.path.join(HERE, f"vc4b_formula_{L.tag()}.json"))


if __name__ == '__main__':
    main(sys.argv[1])

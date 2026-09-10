"""Cross-build scan behind the DECISION tests of
``tests/unit/test_verify_branch_cut_round2.py``: for each of the three
fixtures those gates use, the PRE (engineered) and POST readings over the
truncation ladder, so each bar can be set with a measured two-sided gap on
BOTH builds rather than on the one I happened to develop on.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

sys.path.insert(0, str(VC.TREE / "tests" / "unit"))
import test_verify_branch_cut_round2 as T  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "v9.json"
LAD = (3, 4, 5)


def run():
    p = {"lossy_spacer": {}, "modulation": {}, "detune": {}}
    for M in LAD:
        post = abs(T._closure(T._stack(n_orders=M, spacer_loss=1e-6)))
        with T._pre_arm():
            pre = abs(T._closure(T._stack(n_orders=M, spacer_loss=1e-6)))
        p["lossy_spacer"][str(M)] = {"post": post, "pre": pre,
                                     "ratio": pre / max(post, 1e-300)}
    for rel in (1e-10, 1e-8, 1e-6, 1e-4):
        row = {}
        for M in LAD:
            post = abs(T._closure(T._stack(n_orders=M, rel=rel)))
            floor = abs(T._closure(T._stack(n_orders=M, rel=rel, detune=1e-2)))
            with T._pre_arm():
                pre = abs(T._closure(T._stack(n_orders=M, rel=rel)))
            row[str(M)] = {"post": post, "floor": floor, "pre": pre}
        p["modulation"][f"{rel:.0e}"] = row
    for M in LAD:
        post = {d: T._closure(T._stack(n_orders=M, detune=d))
                for d in (0.0, 1e-6, 1e-4, 1e-2)}
        with T._pre_arm():
            pre = {d: T._closure(T._stack(n_orders=M, detune=d))
                   for d in (0.0, 1e-6, 1e-4, 1e-2)}
        p["detune"][str(M)] = {
            "post_spread": max(post.values()) - min(post.values()),
            "pre_spread": max(pre.values()) - min(pre.values()),
            "post_max_abs": max(abs(v) for v in post.values()),
            "post": {str(k): v for k, v in post.items()},
            "pre": {str(k): v for k, v in pre.items()}}
    VC.dump(OUT, p)
    st = VC.stamp()
    print(f"BUILD py {st['python']} numpy {st['numpy']}")
    print("\nlossy spacer / lossless cell (Im eps_spacer = 1e-6):")
    for M, r in p["lossy_spacer"].items():
        print(f"  M={M}  post {r['post']:.4e}  pre {r['pre']:.4e}  "
              f"ratio {r['ratio']:.3f}")
    print(f"  WORST ratio over the ladder: "
          f"{max(r['ratio'] for r in p['lossy_spacer'].values()):.3f}")
    print("\nmodulation ladder (post / floor / pre) per truncation:")
    for rel, row in p["modulation"].items():
        for M, r in row.items():
            print(f"  rel={rel} M={M}  post {r['post']:.3e}  "
                  f"floor {r['floor']:.3e}  pre {r['pre']:.3e}  "
                  f"pre/floor {r['pre']/max(r['floor'],1e-300):.2e}")
    print("\ndetune spreads:")
    for M, r in p["detune"].items():
        print(f"  M={M}  post_spread {r['post_spread']:.4e}  "
              f"pre_spread {r['pre_spread']:.4e}  ratio "
              f"{r['pre_spread']/max(abs(r['post_spread']),1e-300):.3e}")


if __name__ == "__main__":
    run()

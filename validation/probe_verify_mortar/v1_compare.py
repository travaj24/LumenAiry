"""Compare two v1_bit_identity JSON arms hash-by-hash."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "with"
    b = sys.argv[2] if len(sys.argv) > 2 else "without"
    A = json.load(open(os.path.join(HERE, f"v1_bit_identity_{a}.json")))
    B = json.load(open(os.path.join(HERE, f"v1_bit_identity_{b}.json")))
    print(f"{a}: {A['lumenairy']}  v{A['version']}  py{A['python']} "
          f"np{A['numpy']}")
    print(f"{b}: {B['lumenairy']}  v{B['version']}  py{B['python']} "
          f"np{B['numpy']}")
    total = bad = 0
    for sect in ("basis", "pencil", "stack"):
        if sect not in A or sect not in B:
            print(f"  [{sect}] MISSING in one arm")
            continue
        ka, kb = set(A[sect]), set(B[sect])
        if ka != kb:
            print(f"  [{sect}] KEY SET DIFFERS: only-{a}={sorted(ka-kb)} "
                  f"only-{b}={sorted(kb-ka)}")
        n = m = 0
        for k in sorted(ka & kb):
            n += 1
            if A[sect][k] != B[sect][k]:
                m += 1
                print(f"    MISMATCH {sect}.{k}\n      {a}={A[sect][k]}"
                      f"\n      {b}={B[sect][k]}")
            if str(A[sect][k]).startswith("ERROR"):
                print(f"    ERROR-ARM {sect}.{k}: {A[sect][k]}")
        print(f"  [{sect}] {n - m}/{n} identical")
        total += n
        bad += m
    print(f"TOTAL {total - bad}/{total} identical  "
          f"-> {'BIT-IDENTICAL' if bad == 0 else 'MISMATCHES: %d' % bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Print the jobs of the given jobs files whose output JSON does not exist
yet (for re-queueing after a runner was stopped).

usage: python missing_jobs.py jobs_a.txt [jobs_b.txt ...] > jobs_requeue.txt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D8_TAG = "t{:.4f}_p{:.4f}"


def output_of(cmd):
    a = cmd.split()
    s, mode = a[0], a[1]
    if s == "d2_film.py":
        return f"d2_film_{a[2]}_{a[3]}_{a[4]}.json"
    if s == "d4_pillar.py":
        if mode == "curved":
            return f"d4_curved_{a[2]}_M{a[3]}.json"
        if mode in ("rcwa", "rcwa_scalar"):
            return f"d4_{mode}_n{a[2]}.json"
        if mode == "stair":
            return f"d4_stair_k{a[2]}_M{a[3]}.json"
    if s == "d6_magnetic.py":
        if mode == "stair":
            return f"d6_stair_k{a[2]}_M{a[3]}.json"
        return f"d6_{mode}_{a[2]}_M{a[3]}.json"
    if s == "d5_li2003.py":
        return f"d5_li_{mode}_M{a[2]}.json"
    if s == "d8_oblique.py":
        mat = a[6] if len(a) > 6 else "lc"
        tg = D8_TAG.format(float(a[4]), float(a[5]))
        return f"d8_{mat}_{a[2]}_{tg}_M{a[3]}.json"
    if s == "d7_stack.py":
        return f"d7_{mode}_M{a[2]}.json"
    return None


for fn in sys.argv[1:]:
    for ln in open(os.path.join(HERE, fn)):
        ln = ln.strip()
        if not ln:
            continue
        out = output_of(ln)
        if out is None or not os.path.exists(os.path.join(HERE, out)):
            print(ln)

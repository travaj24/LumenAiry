"""V8 -- BUILD_E1 numbers against the builder's own JSON (spot check).

Each row: (claim in the doc, the JSON reading, relative agreement at the
doc's two significant figures).  Output v8_docs.json."""
import json
import os

import _ve1common as V

B = os.path.join(V.HERE, "..", "build_e1")


def J(name):
    return json.load(open(os.path.join(B, name)))


def row(d, M):
    return [r for r in d["rows"] if r["M"] == M][0]


checks = []


def chk(label, claimed, got):
    ok = abs(got - claimed) <= 0.051 * abs(claimed) + 1e-300
    checks.append(dict(label=label, claimed=claimed, json=got, ok=ok))


c = J("e1_compare.json")
chk("E1-1 200/200 identical", 200, c["n_identical"])
d = J("e3_slab_shear_nonrec_conical.json")
for M, (a, b, e) in {5: (1.9e-10, 9.2e-11, 3.1e-10),
                     7: (8.2e-14, 3.4e-14, 7.1e-14)}.items():
    r = row(d, M)
    chk(f"E1-3 shear conical M{M} dRT", a, r["dRT"])
    chk(f"E1-3 shear conical M{M} dJr", b, r["dJr"])
    chk(f"E1-3 shear conical M{M} dJt", e, r["dJt"])
d = J("e3_slab_s05g3_nonrec_oblique.json")
r = row(d, 8)
chk("E1-3 s05g3 oblique M8 dJt", 1.4e-11, r["dJt"])
d = J("e3_slab_s15g3_nonrec_oblique.json")
chk("E1-3 s15g3 oblique M8 dJt", 1.5e-9, row(d, 8)["dJt"])
d = J("e3_slab_s15g3_lc_oblique.json")
chk("E1-3 s15g3 LC oblique M8 dJt", 3.2e-9, row(d, 8)["dJt"])
a = J("e3_arms_shear_nonrec_normal_M6.json")
chk("E1-3 arms shear normal correct dRT", 6.5e-14, a["correct"]["dRT"])
chk("E1-3 arms shear normal +1j dJr", 1.4e-2, a["hgauge_plus_i"]["dJr"])
chk("E1-3 arms shear normal +1j dJt", 0.40, a["hgauge_plus_i"]["dJt"])
a = J("e3_arms_shear_nonrec_conical_M6.json")
chk("E1-3 arms shear conical rot+ dJt", 0.12, a["rot_flip"]["dJt"])
chk("E1-3 arms shear conical rot+ dRT", 1.7e-3, a["rot_flip"]["dRT"])
d = J("e3_slab_c3_nonrec_normal.json")
chk("E1-4 c3 normal M8 dRT", 8.9e-14, row(d, 8)["dRT"])
chk("E1-4 c3 normal M6 dJt", 1.3e-10, row(d, 6)["dJt"])
d = J("e3_slab_c5_nonrec_normal.json")
chk("E1-4 c5 normal M6 dRT", 1.4e-13, row(d, 6)["dRT"])
e5 = J("e5_summary.json")
chk("E1-5 c5 M7 vs c3 M10", 4.0e-6, e5["c3_top_vs_c5_top"])
chk("E1-5 RCWA Richardson (25,29)", 7.0e-5,
    e5["rcwa_richardson_1_over_N"][-1]["to_c3_top"])
chk("E1-5 c5 M4 to c3 top", 3.2e-4, e5["c5"][0]["to_c3_top"])
chk("E1-5 stair k1 M4", 4.5e-2, e5["stair"]["1"][0]["to_c3_top"])
e6 = J("e6_summary.json")
chk("E1-6 c5 M7 vs c3 M10", 2.3e-5, e6["c3_top_vs_c5_top"])
chk("E1-6 zstair (4,8)", 2.2e-4, e6["zstair_limit_1_over_Nz2"][0]["to_c3_top"])
chk("E1-6 zstair (8,16)", 2.6e-4, e6["zstair_limit_1_over_Nz2"][1]["to_c3_top"])
chk("E1-6 stair k1 M10", 7.6e-2, e6["stair"]["1"][-1]["to_c3_top"])
chk("E1-6 c5 M4", 6.3e-4, e6["c5"][0]["to_c3_top"])
e7 = J("e7_summary.json")
chk("E1-7 slanted recip M7", 1.5e-6, e7["slanted"]["7"]["reciprocal_control"])
chk("E1-7 slanted nonrec M7", 5.5e-3, e7["slanted"]["7"]["nonrec_vs_same"])
chk("E1-7 slanted transposed M7", 1.4e-6,
    e7["slanted"]["7"]["nonrec_vs_transposed"])
chk("E1-7 slanted nonrec M5", 6.2e-3, e7["slanted"]["5"]["nonrec_vs_same"])
e9 = J("e9_summary.json")
chk("E1-9 pillar no_mu_blocks dRT", 7.1e-2, e9["pillar"]["no_mu_blocks"]["dRT"])
chk("E1-9 pillar rot_flip dRT", 4.8e-2, e9["pillar"]["rot_flip"]["dRT"])
c8 = J("e10_cost_M8.json")
chk("E1-10 inplane mapped M8 total", 22.8, c8["inplane_mapped"]["total"])
chk("E1-10 oop mapped M8 total", 18.5, c8["oop_mapped"]["total"])
chk("E1-10 oop mapped peak MB", 501, c8["oop_mapped"]["peak_MB"])
chk("E1-10 oop slant mapped total", 13.5, c8["oop_slant_mapped"]["total"])
nok = sum(1 for x in checks if not x["ok"])
V.dump("v8_docs.json", {"checks": checks, "n": len(checks), "mismatch": nok})
for x in checks:
    print(("OK  " if x["ok"] else "MISS"), x["label"], x["claimed"],
          f"{x['json']:.3g}")
print("checks", len(checks), "mismatch", nok)

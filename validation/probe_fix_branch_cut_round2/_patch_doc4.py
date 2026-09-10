"""One-shot editor: add sections 4.7 (the interface conditioning) and 4.8 (the
manufactured-energy mount) to the round-2 report."""
import io

NEW = r"""### 4.7 The sharp instrument: `cond(a + b)` on the PMM's own interfaces

Round 1 localised the RCWA defect not by an accuracy reading but by the
CONDITIONING of the interface mode-match, and that instrument is what settles
the mechanism here too, because it is a property of the operator rather than of
the truncation error.  `b2_pmm_interface.py` patches every PMM binding of
`_interface_smatrix` and records `cond(a + b)` at each mode match (WIN, 1
thread; `inst` is the shipped round-2 code, `pre` the reinstated exact-zero pin):

| fixture | class | `cond(a+b)` PRE | POST |
|---|---|---|---|
| `cell_weak_region_M3` | region | **6.556e+08** | 1.978e+01 |
| `cell_weak_none_M3` | none | 9.918e+01 | 1.978e+01 |
| `cell_weak_region_M5` | region | **9.080e+05** | 6.879e+01 |
| `cell_weak_region_M7` | region | **6.802e+07** | 1.244e+02 |
| `cell_weak_region_M9` | region | **6.486e+08** | 2.031e+04 |
| `cell_weak_superstrate` | superstrate | **1.173e+07** | 5.094e+01 |
| `hyb_weak_spacer_M3` | spacer | **1.959e+09** | 1.453e+01 |
| `hyb_weak_both_M3` | both | **1.959e+09** | 1.453e+01 |
| `hyb_weak_both_oblique` | both | **1.920e+08** | 1.527e+01 |
| `stag_weak_region` | region (staggered) | 1.539e+02 | 1.539e+02 |
| `pure_weak_both` | both (staggered) | 6.590e+01 | 6.590e+01 |

Seven to nine decades on the coincident mounts, nothing on the off-coincidence
control, nothing at all on the staggered engine.  The `cell_weak_superstrate`
row is the SUPERSTRATE-side coincidence (`n_superstrate^2 = 2.25`,
`n_substrate = 1.9`), which behaves exactly as the substrate side does.

### 4.8 The loudest mount: the pre-round-2 hybrid PMM MANUFACTURES energy

The 4.2 fixture samples the cell on a 6-pixel grid.  Sampling the SAME device on
a 32-pixel grid -- three strips per axis instead of three coarse cells, an
identical piecewise-constant permittivity -- makes the pre-round-2 error three
decades larger.  `b10_manufactured_energy.py`, `n_orders` 2..6 with and without
the spacers, plus a spacer-detune ladder, on both builds at 1 / 4 / 8 threads:

| build / threads | worst PRE `sum R + T` | worst PRE per-order motion | worst PRE NO-SPACER control | worst POST (all mounts) |
|---|---|---|---|---|
| WIN 1 | **5.812454299** (2.9x) | 2.788e+00 | 8.8539e-10 | 8.8539e-10 |
| WIN 4 | **1.098858e+02** (55x) | 5.755e+01 | 8.8539e-10 | 8.8539e-10 |
| WIN 8 | **3.567077e+01** (18x) | 4.674e+00 | 8.8539e-10 | 8.8539e-10 |
| WSL 1 | 1.999336151 | 2.076e-02 | 8.8540e-10 | 8.8539e-10 |

A passive lossless stack returning 110 times the incident power is not a
tolerance question.  Five things this adds.

* **The POST envelope and the PRE NO-SPACER control are the SAME number**,
  8.8539e-10, on every one of the eight (build, thread, arm) samples.  It is the
  `n_orders = 4` control mount's own Fourier truncation error -- deterministic,
  arm-independent, and the floor every bar in the round-2 test file is derived
  against.
* **Five decades of spread with the BLAS thread count** on the PRE arm
  (6.6e-04 .. 1.1e+02), and a per-build partition of WHICH truncation breaks:
  Windows breaks `n_orders` 3 and 4, WSL breaks 5 and 6.  That is the branch-cut
  signature, one level up.
* **It is not always loud.**  At `n_orders = 3` Windows emits a `UserWarning`
  (`sum R + T = 4.14` is past the energy tripwire), but the spacer detuned by a
  relative 1e-6 returns `sum R + T = 2.000072107` **SILENTLY**, with per-order
  efficiencies 9.331e-05 away from the repaired answer.  A caller who followed
  the library's own "detune by ~1e-6" advice would have got a quiet wrong
  answer.
* **The no-spacer control never moves**, on either arm, at any truncation or
  thread count.  The spacer is the cause, isolated.
* **Every POST mount reads 2.000000000** to nine decimals.

The gate `test_the_coincident_spacer_stack_does_not_manufacture_energy` asserts
on the LADDER rather than on one mount, precisely because which truncation
breaks is a per-build fact.

"""

p = "docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md"
s = io.open(p, encoding="utf-8").read()
anchor = "### 4.6 Is the repaired answer RIGHT?"
i = s.index(anchor)
j = s.index("## 5. The band scale")
# keep 4.6 then insert 4.7/4.8 before the section-5 rule
tail = s[i:j]
sep = "\n---\n\n"
assert tail.endswith(sep), repr(tail[-20:])
io.open(p, "w", encoding="utf-8", newline="").write(
    s[:i] + tail[:-len(sep)] + NEW + sep + s[j:])
print("sections 4.7/4.8 added")

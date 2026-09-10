"""One-shot editor: add section 5.5 (the staggered selector's own band, a
cross-check on a function round 2 does not change) to the round-2 report."""
import io

NEW = r"""### 5.5 A cross-check that is NOT this function's: `_forward_branch_flip`

Round 1 adopted `1e-8` from `pmm/_core._forward_branch_flip`, calling it "one
convention for one quantity across the two solvers".  The verification already
qualified that ("both are `1e-8 * max(max|.|, 1)`, but they threshold DIFFERENT
quantities and flip differently"), and measuring the staggered population makes
the qualification concrete: **the two selectors use OPPOSITE conventions.**

* `_sqrt_decay` takes `lam^2`, and a PROPAGATING mode has it real NEGATIVE (on
  the principal square root's cut), so its discriminator is the ROOT's REAL
  part;
* `_forward_branch_flip` takes `q = kz/k0 = sqrt(gamma^2)`, and a PROPAGATING
  mode has `q` real, i.e. `gamma^2` real POSITIVE, so its discriminator is
  `|Im(q)|` and its rule is
  `flip <=> Im(q) < -tol or (|Im(q)| <= tol and Re(q) < 0)`.

Scoring the staggered pencil with `_sqrt_decay`'s ratio -- which is what a
single "census every population the same way" pass does -- therefore measures
the wrong thing, and `b6_band_scale.py`'s `pmm_staggered` row (a 1.67-decade
gap) is that mistake.  `b6c_staggered_flip.py` re-measures it on the quantity
the selector actually thresholds, over 116 eigenvalue arrays and 42,310 modes
from the staggered 2-D engine (scalar, tensor, oblique, lossy, the pure stack)
and the 1-D PMM (TE, TM, conical, a loss ladder):

| classification of "the imaginary part is ROUNDING" | noise side max | signal side min | gap |
|---|---|---|---|
| `\|Im g2\| <= 1e-6 \|Re g2\|` (the `_sqrt_decay` census convention) | 1.5509e-08 | 2.4337e-08 | 0.20 dec |
| `\|Im g2\| <= 1e-12 \|Re g2\|` | **1.2160e-15** | 1.5053e-12 | 3.09 dec |

Neither classification is satisfying, and the reason is instructive: on THIS
convention a relative `1e-6` on `gamma^2` admits modes whose `|Im q| / |q|`
reaches 5e-7 -- six decades above the eigensolver's backward error -- while the
`1e-12` one calls "signal" a population of nearly-propagating modes whose
`Im(q)` is a negligible fraction of `Re(q)`, exactly the modes for which the
`Re(q) < 0` rule is the right discriminator.  The natural quantity for that
selector is `|Im q| / |q|`, not `|Im q| / max|q|`.

What CAN be said cleanly, and is the reason this is a cross-check rather than a
finding: **the worst `|Im q| / |q|` that `_forward_branch_flip` actually acts on
over this whole population is 3.5588e-09.**  Every mode the band reaches has an
imaginary part at most 3.6e-09 of its own magnitude, i.e. unambiguously
rounding, so it never re-signs a physical decay rate here.

`_forward_branch_flip` is not changed by round 2 and no claim about it is made
beyond that.  Whether its scale should be per-mode -- which is a different
question from section 5.3's, because its convention is the other one -- is
recorded as an open item in section 11.

---

"""

p = "docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md"
s = io.open(p, encoding="utf-8").read()
anchor = "## 6. X-1 is CLOSED"
i = s.index(anchor)
io.open(p, "w", encoding="utf-8", newline="").write(s[:i] + NEW + s[i:])
print("section 5.5 added")

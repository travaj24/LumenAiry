# WP-C3 round 3 -- the focus readout's standoff leg takes the Collins transport

Evidence for the maintainer decision of 2026-09-20 (`MAINTAINER_DECISIONS_2026_09.md`
section 0.8; `fixes/WP-C3_COLLINS_DEFAULT_REPORT.md` section "Round 3").
Every JSON here records `lumenairy.__file__`, the interpreter, NumPy's version
and the readout's live `transport` default, so no row can be attributed to the
wrong tree.

## The probes

| file | what it measures |
|---|---|
| `r3_a6_both_transports.py` | the fixtures behind the eight a6 ids, on BOTH transports: the resolved and the pre-fix (carrier-only) standoff legs, their focal peaks and piston-free relative L2 against the analytic Gaussian-ABCD field, the containment the guard read, and the default disposition |
| `r3_oracle_and_entrypoints.py` | (a) the matched row and the closed-form focal peak `(w_in/w0)^2`; (b) the downward-quadratic fixture's gated leg on both transports; (c) **the oracle certification** -- the analytic ABCD width against the exact second-moment law `<r^2>(z) = <r^2> + 2 z <r.theta> + z^2 <theta^2>`, the law read off the ENVELOPE with the carrier composed in closed form so nothing differentiates an under-sampled phase; (d) an AST census of every call site of the readout in the shipped package, and the GUI |

Both take `<tree> <out.json>` and refuse a tree other than the one named.

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=<tree> python r3_a6_both_transports.py <tree> out.json
```

## The readings

* **oracle certification** -- worst relative difference between the closed
  form and the second-moment law **1.216626996434166e-13** over 15 cells
  spanning three fixtures and `z/|R|` from 0.02 to 0.97, identical on
  WIN-py3.14 and WSL-py3.12.  That is what makes the closed form usable AS
  the law.
* **the pre-fix leg** -- focal peak as a fraction of the analytic one:
  co-moving 0.745373 / 0.187898 / 0.026307 at `R/R0` = 0.98 / 0.95 / 0.90,
  Collins 0.997221 / 0.997137 / 0.995544.  Both builds agree to 3.3e-16.
* **the downward-quadratic fixture** -- resolved leg relL2 3.5794e-04
  (co-moving) against 2.6062e-06 (Collins); the gated (pre-fix) leg
  1.0022e-01 against 9.7717e-07, with the co-moving arm refused by the guard
  at containment 0.866216 and the Collins arm returning at 2.840270.
* **entry points** -- exactly two internal call sites, `carrier.py` 11705 and
  11743, both inside `propagate_traced_carrier_chain` and both naming
  `transport='sziklas'`; no module under `lumenairy/ui/` reaches the readout.

## The censuses

`blastwidth_r3_{pre,flip}_{win,wsl}.json` are the 192-cell ordinary-chain
sweep (`validation/probe_verify_c3/probe_vc3_blastwidth.py`, unmodified) on
`git archive wave5/with-c3` and on this branch: **192 / 0 / 0** either way,
both builds, 0 Kelly warnings.

`wayback_r3_{win,wsl}_summary.json` are the 103-key archive-to-archive
comparisons (`validation/probe_verify_c3/probe_wayback.py`, unmodified):
`pre:mine` **98 identical / 5 moved**, 0 ok->raised, 2 raised->ok, on both
builds; `waypre:way` (both trees with `transport='sziklas'`) **103 identical**
on both builds, recorded for Windows in
`wayback_r3_identity_win_compare.json` and for WSL inside the summary.

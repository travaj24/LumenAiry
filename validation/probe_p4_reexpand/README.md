# probe_p4_reexpand -- ITEM 1 of `docs/audits/FIX_P4_TRACED_CLOSEOUT_2026_08_24.md`

Adjudicating S14.6's one unreproduced failure of
`test_niche_p4_gbd_reexpand.py::test_frame_completeness_metric_published`.

```sh
export LUMENAIRY_ROOT=/path/to/this/checkout
python census_solves.py 2      # which least-squares solves the P4 path makes (S1.2)
python statefree.py            # 868-binding module-state diff across the call (S1.4)
python route.py                # reconstruct-route decision margins (S1.6)
sh run_wave1.sh 60 20000       # 6 pair arms + 2 fit hammers, concurrent (S1.7)
sh run_wave2.sh 60 20          # 5 pair arms at widths 1/2/4/8/16 + a threads arm
sh run_interleaved.sh 3        # the A/B/A/B load control (S1.7.1)
```

Every arm writes one JSONL line per iteration carrying the pair verdict, both
field hashes, `max|E_a - E_b|` and every bar the test asserts.  The committed
logs are those runs.  `wave1_fit*.jsonl` (2 x 20 000 lines, 8.6 MB) is reduced
to `wave1_fit_summary.json`; `stress_fit.py` re-captures its `(A, b)` on demand
if `fit_AB.npz` is absent.

Result: **782 pairs, 0 mismatches, one field hash `18c25df511ad134f`**, plus
40 000 solves of the path's only BLAS-adjacent step at one coefficient hash.

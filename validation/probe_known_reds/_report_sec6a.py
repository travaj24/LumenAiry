"""Fill in the report's section 6 as CI work items land."""
import io
import sys

P = ('docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/'
     'WP-B14_KNOWN_REDS_REPORT.md')

OLD = """## 6. The CI matrix (run 34914295323)

_This section is completed in section 6.x below as each work item lands._
"""

NEW = '''## 6. The CI matrix (run 34914295323)

34 jobs, 30 red, on the 5.47.0 release commit.  The matrix now runs Python
3.10-3.14 x 5 shards, a 5-shard slow lane, a JAX job and a strict mypy job.
Downloaded logs: `C:/tmp/ci_5470/<job>/`.  Job-level outcome as found:

| lane | outcome as found |
|---|---|
| py3.10 x 5 | **every shard aborted at collection** -- `18 skipped, 1 error`; the lane tested nothing |
| py3.11 x 5 | **every shard aborted at `--maxfail=10`** -- 10 failed / 178-223 passed each |
| py3.12 x 5, py3.13 x 5, py3.14 x 5 | 1-8 failures per shard |
| slow lane x 5 | **all five timed out** at the 30-minute step cap (1847-1863 s each; shard 2 reached 65 %, shard 5 reached 98 %, no summary line) |
| mypy strict | 1 error |
| JAX | 2 failures, both already in the classes above |

Because 3.11 aborted at 10 per shard, **the 3.11 failure set is a lower bound**
and the census of what is broken on 3.11 is unknowable from that run -- which is
itself the argument for the `--maxfail` item below.

### 6.1 py3.10 -- the whole lane aborted at collection

`tests/unit/test_audit2609_a15a_packaging.py` imports `tomllib`
unconditionally; `tomllib` is 3.11+, and CI installs `tomli` on 3.10 for exactly
this reason (the install step says so).  Fixed at the test layer with the
standard fallback, so the lane runs at all.  Consequence worth stating: **py3.10
has been running zero of the 2 800-odd ids per shard**, so the first green 3.10
lane will be the first time several gates -- the a17 history gates among them --
are exercised on that interpreter.

### 6.2 py3.11 -- the a17 history token digest (PEP 701)

**49 distinct ids** of
`test_the_module_token_stream_is_unchanged_since_the_history_move` failed on
3.11 and **0** on 3.12 / 3.13 / 3.14.  The sibling AST test was **123/123 green
on 3.11**, which is the first hard datum: the module sources are identical on
every arm, so the thing that moved with the interpreter is the digest's own
definition.

Mechanism, measured: PEP 701 landed in 3.12, so `x = f"a{b}c"` tokenises as
**seven** records on 3.12+ (`FSTRING_START` / `FSTRING_MIDDLE` / `OP` / `NAME` /
`OP` / `FSTRING_MIDDLE` / `FSTRING_END`) and **one** `STRING` record before it.
**110 of the 123** registered modules contain an f-string (3 445 `JoinedStr`
nodes); **13** do not; the 7 ids that PASSED on 3.11 are all in the 13 and the
49 that failed are all in the 110, both directions clean.  So the true count of
affected modules is **110**, not the 49-id lower bound the aborted run showed.

The decisive step, taken without a 3.11 interpreter on the box: the actual
3.11-computed digests were scraped out of the 49 CI failure blocks, and the new
scheme -- collapsing each `FSTRING_START..FSTRING_END` run to one `STRING`
record carrying the f-string's exact **source slice** -- reproduces **49 of 49**
of them bit-for-bit on 3.12.3, 3.13.13 and 3.14.6 alike.  That leaves no
residual for any other tokenizer difference (NEWLINE/NL/INDENT sequencing and
the rest), so the mechanism is established rather than assumed.

Fixed by making the digest version-independent (the brief's option A), not by
skipping below 3.12, because the claim the test exists for **can** be kept: the
source slice is FINER than the token run (which normalises `{{` to `{` and says
nothing about spacing inside a replacement field).  Six new falsifiability cases
prove the digest still moves on a value-preserving re-spelling of an f-string --
quote character, prefix case, spacing inside `{...}`, a conversion, a format
spec, an implicit concatenation -- and three companions prove the first three
are invisible to the AST fingerprint, which is what keeps the two fingerprints
independent.  A registry-wide sweep asserts no tokenizer-specific record name
ever reaches the digest, which is the guard that catches the next PEP 701.

**110 of 123** history documents re-recorded by the recorder (never by hand),
each exactly +2/-1 lines; `ast_sha256` moved on **0 of 123**, which is the
arithmetic proof that this was a digest-scheme change and not a code change.
`record_history_fingerprints.py --check` is green, and the recorder and the gate
were confirmed to share ONE implementation of the digest (no second copy to
drift).

Kernel ladder 8/8 arms: 123/123 both fingerprints, rc=0 -- no build dependence,
as expected for a digest that touches neither numpy nor BLAS.  The dependence
was on the interpreter alone.

### 6.3 The a8 glass tests -- the library was right, the tests were not

Seven CI reds, all in `test_audit2609_a8_glass.py` and
`test_audit2609_a8_verify.py`, from CI deliberately not installing the glass
extra.  The brief's hypothesis was that the first two --
`get_glass_index_complex raised ... instead of falling back to kappa = 0` --
were a LIBRARY contract violation.  **They are not, and `lumenairy/glass.py` is
byte-identical to the base commit** (md5 checked both sides).

The reasoning is worth keeping.  `refractiveindex` IS installed on this box, so
its absence had to be made a FIXTURE: a blocker that wraps
`importlib.util.find_spec` and sets `sys.modules['refractiveindex'] = None`
before `lumenairy.glass` is first imported (the module decides availability once
at import).  Against a pristine `git archive` tree that fixture reproduces
**exactly the seven CI ids and nothing else**.

With the package blocked, of 49 tuple-registered glasses **exactly one --
`SILICON` -- has no bundled row**; the other 48 return `n + 0j` with one
warn-once each.  And the `ImportError` does not come from the extinction path at
all: `get_glass_index` itself cannot produce a real index for `SILICON`, and
`get_glass_index_complex` reaches `kappa` only after the real index.  Catching
it and "falling back to kappa = 0" would mean **fabricating a real index** --
precisely the silent-wrong shape WP-A8's own E2 second arm exists to kill.  The
module docstring settles it ("Only the tuple-style entries that lack a Sellmeier
fallback will raise"), and so does an internal contradiction: a sibling test in
the same family REQUIRES `get_glass_index_complex` to raise `ValueError` out of
page range, so "never raises" was never literal.

`pytest.importorskip` was ruled out for all of them -- not by preference but by
`docs/TESTING_STANDARDS.md` rule 4 ("Never `pytest.skip` on a resource check --
two skips silently removed five tests from the gate on exactly the runners that
mattered"), which both files already cite.  Every test instead asserts the
documented no-package fact as a two-sided partition: the exempt set must EQUAL
the independently computed "tuple entry with no bundled row on this install"
set, each exemption must be an `ImportError` naming both remediations, the
premise is re-measured, and a hard floor keeps the sweep from shrinking.
`test_e2_missing_kappa_warns_once_and_returns_zero` is split PER PARAMETER, so
the three glasses that do exercise the fallback still run on both installs; a
blanket skip would have deleted them.
`test_e2_pages_that_do_carry_k_keep_their_value_and_sign` is split by half --
the real-index half is bit-identical on both installs
(`0x1.802abb7771dacp+0`) and stays unconditional.

Not weakened: with the package present, restoring the pre-WP-A8 catch tuple
turns the rewritten gate red naming the right glasses, so the exemption cannot
swallow an extinction regression.

```
a8_glass + a8_verify, package PRESENT : 64 passed, 0 skipped
a8_glass + a8_verify, package BLOCKED : 64 passed, 0 skipped
```
Same collection count on both arms, no coverage deleted.
'''


def main():
    with io.open(P, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    if src.count(OLD) != 1:
        print('anchor not unique:', src.count(OLD))
        return 1
    src = src.replace(OLD, NEW)
    with io.open(P, 'w', encoding='utf-8', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('section 6 written')
    return 0


if __name__ == '__main__':
    sys.exit(main())

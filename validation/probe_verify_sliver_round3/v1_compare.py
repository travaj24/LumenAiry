"""V1-COMPARE -- score two ``v1_bitid.py`` JSONs against each other.

    python v1_compare.py BEFORE.json AFTER.json out.json

BEFORE is the round-3 branch point ``f2371e0`` (round 2 + its verification);
AFTER is the tree that carries round 3.

What is scored
--------------
1. **Bit-identity of the RETURNED answer.**  Every fixture returned by both
   trees must carry the same SHA-256 of the raw ``R``, ``T`` and Jones
   buffers.  A single differing digest is a broken number, not a tolerance.
2. **The warning set.**  Compared verbatim after ONE normalisation: the
   ``truncation`` note is excised from each warning (it is the sentence round
   3 rewrote), and what is left must match character for character.  Whether
   each fixture carried the note is compared as a flag.
3. **The refusals are a superset.**  Round 3 claims a strict WIDENING, so
   every fixture BEFORE refused must still be refused, and no fixture may go
   the other way.
4. **The flips, classified independently.**  Every fixture whose decision
   moves is scored by two references this probe measured itself: the exact
   ``delta -> 0`` solve of the same device (``kind``) and the solve on the
   PRESCRIBED ``min_feature`` grid (``kind_snap``, plus the distance between
   the returned answer and that grid's answer in units of the widest
   manufactured cell).
"""
import json
import sys

MARK = "  A near-coincident-wall SLIVER ("


def strip_note(w):
    """The warning with the round-3 ``truncation`` note removed."""
    i = w.find(MARK)
    return (w if i < 0 else w[:i]), (i >= 0)


def norm(warns):
    out, notes = [], []
    for w in warns:
        base, has = strip_note(w)
        out.append(base)
        notes.append(has)
    return out, notes


def main():
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    b = json.load(open(sys.argv[2], encoding="utf-8"))
    dest = sys.argv[3] if len(sys.argv) > 3 else "v1_compare.json"
    A = {r["name"]: r for r in a["rows"]}
    B = {r["name"]: r for r in b["rows"]}
    assert set(A) == set(B), (set(A) ^ set(B))

    bit_ok = bit_bad = 0
    raw_ok = raw_bad = 0
    warn_ok = warn_bad = 0
    note_added = note_removed = 0
    flips = []
    msg_same = msg_diff = 0
    both_refused = 0
    for nm in sorted(A):
        ra, rb = A[nm], B[nm]
        # (1) the UNGUARDED solve must be identical too -- the change is in
        #     the decision, not in the arithmetic
        if (ra["raw_R"], ra["raw_T"], ra["raw_J"]) == (rb["raw_R"],
                                                       rb["raw_T"],
                                                       rb["raw_J"]):
            raw_ok += 1
        else:
            raw_bad += 1
        if not ra["refused"] and not rb["refused"]:
            if (ra.get("ret_R"), ra.get("ret_T"), ra.get("ret_J")) == (
                    rb.get("ret_R"), rb.get("ret_T"), rb.get("ret_J")):
                bit_ok += 1
            else:
                bit_bad += 1
            wa, na = norm(ra["warns"])
            wb, nb = norm(rb["warns"])
            if wa == wb:
                warn_ok += 1
            else:
                warn_bad += 1
                flips.append(dict(name=nm, kind="WARNING-SET",
                                  before=wa, after=wb))
            if na != nb:
                note_added += sum(1 for x, y in zip(na, nb) if y and not x)
                note_removed += sum(1 for x, y in zip(na, nb) if x and not y)
        if ra["refused"] and rb["refused"]:
            both_refused += 1
            if ra["msg"] == rb["msg"]:
                msg_same += 1
            else:
                msg_diff += 1
        if ra["refused"] != rb["refused"]:
            flips.append(dict(
                name=nm, kind=("truncation->sliver" if rb["refused"]
                               else "sliver->truncation"),
                delta=ra.get("delta"), worst=ra.get("worst"),
                su_snap=ra.get("su_snap"), drop=ra.get("drop"),
                move_w=ra.get("move_w"),
                arb_before=ra.get("arb"), arb_after=rb.get("arb"),
                err_d=ra.get("err_d"), kind_returned=ra.get("kind"),
                err_snap_d=ra.get("err_snap_d"),
                kind_snapped=ra.get("kind_snap")))
    doc = dict(
        before=a["build"], after=b["build"],
        n=len(A),
        returned_by_both=bit_ok + bit_bad,
        returned_bit_identical=bit_ok, returned_bit_broken=bit_bad,
        unguarded_identical=raw_ok, unguarded_broken=raw_bad,
        warning_sets_identical_after_note_excision=warn_ok,
        warning_sets_differing=warn_bad,
        truncation_notes_added=note_added,
        truncation_notes_removed=note_removed,
        refused_before=sum(1 for r in a["rows"] if r["refused"]),
        refused_after=sum(1 for r in b["rows"] if r["refused"]),
        refused_by_both=both_refused,
        refusal_message_identical=msg_same,
        refusal_message_reworded=msg_diff,
        flips=flips,
        flips_truncation_to_sliver=sum(
            1 for f in flips if f["kind"] == "truncation->sliver"),
        flips_sliver_to_truncation=sum(
            1 for f in flips if f["kind"] == "sliver->truncation"),
        flips_wrong_by_delta0=sum(
            1 for f in flips if f.get("kind_returned") == "wrong"),
        flips_right_by_delta0=sum(
            1 for f in flips if f.get("kind_returned") == "right"),
        flips_snapped_right=sum(
            1 for f in flips if f.get("kind_snapped") == "right"),
        mildest_flip_err_d=min(
            [f["err_d"] for f in flips if f.get("err_d") is not None],
            default=None),
    )
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    for k in ("n", "returned_by_both", "returned_bit_identical",
              "returned_bit_broken", "unguarded_broken",
              "warning_sets_identical_after_note_excision",
              "warning_sets_differing", "truncation_notes_added",
              "truncation_notes_removed", "refused_before", "refused_after",
              "refused_by_both", "refusal_message_identical",
              "refusal_message_reworded", "flips_truncation_to_sliver",
              "flips_sliver_to_truncation", "flips_wrong_by_delta0",
              "flips_right_by_delta0", "flips_snapped_right",
              "mildest_flip_err_d"):
        print(f"{k:52s} {doc[k]}")


if __name__ == "__main__":
    main()

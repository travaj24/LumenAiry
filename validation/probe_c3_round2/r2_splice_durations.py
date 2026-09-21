"""WP-C3 round 2 -- splice freshly measured durations into ``.test_durations``.

VERIFY-WP-C3 D12 recorded that the branch never spliced its 26 new ids (0 of
26 present), which degrades the shard balance the staleness gate exists to
protect.  This re-records, PER FILE, every file round 2 added ids to or
restated ids in: every existing entry for that file is dropped and the
measured ones are inserted, so a RENAMED id leaves no orphan behind.

    python r2_splice_durations.py <durations.json> [<durations.json> ...]

Each input is a pytest-split ``--store-durations`` file measured SERIALLY
with the three thread caps on the command line.
"""
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
TARGET = os.path.join(REPO, '.test_durations')


def main():
    with open(TARGET, encoding='utf-8') as fh:
        cur = json.load(fh)
    n0 = len(cur)
    new = {}
    for p in sys.argv[1:]:
        with open(p, encoding='utf-8') as fh:
            blob = json.load(fh)
        # A LATER input supersedes an earlier one for the files it covers, so
        # a RENAMED id measured in an earlier pass leaves no orphan.
        covers = {k.split('::')[0] for k in blob}
        for k in [k for k in new if k.split('::')[0] in covers]:
            del new[k]
        new.update(blob)
    files = sorted({k.split('::')[0] for k in new})
    dropped = [k for k in cur if k.split('::')[0] in files]
    for k in dropped:
        del cur[k]
    cur.update(new)
    out = {k: cur[k] for k in sorted(cur)}
    # Match the file's existing shape exactly: 2-space indent, sorted keys,
    # CRLF, no trailing newline change.
    with io.open(TARGET, 'w', encoding='utf-8', newline='\r\n') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(f'files re-recorded : {len(files)}')
    for f in files:
        print('   ', f)
    print(f'entries dropped   : {len(dropped)}')
    print(f'entries inserted  : {len(new)}')
    print(f'total             : {n0} -> {len(out)}')
    with open(TARGET, encoding='utf-8') as fh:
        json.load(fh)
    print('VALID JSON')


if __name__ == '__main__':
    main()

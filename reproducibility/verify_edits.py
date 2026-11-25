"""Recount token edits from the included text and compare with the stored counts.

    python verify_edits.py

Runs on the corpora whose text is included (LSVSC and FLEURS). For ViVOS and ViMD only counts are
distributed, so there is nothing to recount.
"""
import json
import sys

from metrics import DATA, levenshtein

if __name__ == '__main__':
    total = wrong = 0
    for path in sorted(DATA.glob('*/*.json')):
        if not path.name.startswith(('main_', 'component_')):
            continue
        rows = json.loads(path.read_text())['utterances']
        n = bad = 0
        for r in rows.values():
            if 'ref' not in r:
                continue
            n += 1
            ref = r['ref'].split()
            if levenshtein(ref, r['hyp'].split()) != r['e'] or len(ref) != r['n']:
                bad += 1
        if n:
            print(f'{path.parent.name}/{path.name}: {n} utterances recounted, {bad} mismatches')
        total += n; wrong += bad
    print(f'TOTAL {total} recounted, {wrong} mismatches')
    sys.exit(1 if wrong else 0)

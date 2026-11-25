"""Metrics and resampling used by the article. Standard library plus NumPy."""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def load(corpus, system):
    return json.loads((DATA / corpus / f'{system}.json').read_text())['utterances']


def levenshtein(a, b):
    """Edit distance between two sequences (used only by verify_edits.py)."""
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            t = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, p + (x != y))
            p = t
    return d[-1]


def wer(rows, ids=None, edits='e'):
    ids = list(rows) if ids is None else ids
    return 100 * sum(rows[u][edits] for u in ids) / sum(rows[u]['n'] for u in ids)


def cer(rows):
    return 100 * sum(r['ec'] for r in rows.values()) / sum(r['nc'] for r in rows.values())


def group(uid):
    """Resampling unit: ViVOS speaker, LSVSC sequential block, ViMD supplied speaker ID."""
    return uid.split('_')[0]


def paired_numpy(a, b, ids=None, seed=20261117, reps=10000):
    """Paired group percentile bootstrap of WER(a) - WER(b), NumPy generator.

    Reproduces the rescoring-versus-Full8 intervals quoted in Section IV-C."""
    ids = list(a) if ids is None else ids
    acc = {}
    for u in ids:
        g = acc.setdefault(group(u), [0, 0, 0])
        g[0] += a[u]['e']; g[1] += b[u]['e']; g[2] += a[u]['n']
    values = np.array([acc[g] for g in sorted(acc)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, reps, 1000):
        idx = rng.integers(0, len(values), size=(min(1000, reps - start), len(values)))
        tot = values[idx].sum(axis=1)
        draws.append(100 * (tot[:, 0] - tot[:, 1]) / tot[:, 2])
    draws = np.concatenate(draws)
    total = values.sum(axis=0)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(difference=float(100 * (total[0] - total[1]) / total[2]), ci95=[float(lo), float(hi)], groups=len(values))


def paired_python(a, b, seed=20261117, reps=10000):
    """Paired group percentile bootstrap of WER(a) - WER(b), Python generator.

    Reproduces the subset-minus-IRT intervals of the component retune table."""
    acc = {}
    for u in a:
        g = acc.setdefault(group(u), [0, 0, 0])
        g[0] += a[u]['e']; g[1] += b[u]['e']; g[2] += a[u]['n']
    keys = list(acc)
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        ea = eb = n = 0
        for k in (rng.choice(keys) for _ in keys):
            x, y, z = acc[k]
            ea += x; eb += y; n += z
        diffs.append(100 * (ea - eb) / n)
    diffs.sort()
    return dict(difference=wer(a) - wer(b), ci95=[diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1]], groups=len(keys))


def eligible(record):
    """Selection pool of the output cases: 12-30 reference tokens and no digit character in the reference.

    Records whose text is withheld carry the precomputed flag `has_digit` instead of the reference."""
    has_digit = record['has_digit'] if 'has_digit' in record else any(ch.isdigit() for ch in record['ref'])
    return 12 <= record['n'] <= 30 and not has_digit


LENGTH_BINS = [(1, 9, '<10'), (10, 19, '10-19'), (20, 39, '20-39'), (40, 79, '40-79'), (80, 10 ** 9, '>=80')]


def by_length(systems, min_utterances=20):
    """Pooled WER per reference-length bin for a dict name -> rows."""
    first = next(iter(systems.values()))
    out = []
    for lo, hi, label in LENGTH_BINS:
        ids = [u for u, r in first.items() if lo <= r['n'] <= hi]
        if len(ids) < min_utterances:
            continue
        row = dict(bin=label, utterances=len(ids), tokens=sum(first[u]['n'] for u in ids),
                   **{k: wer(v, ids) for k, v in systems.items()})
        row['edits'] = {k: sum(v[u]['e'] for u in ids) for k, v in systems.items()}
        if 'Rescue' in row and 'Full8' in row:
            row['Rescue_minus_Full8'] = row['Rescue'] - row['Full8']
            row['relative_reduction_vs_Rescue'] = 100 * (row['Rescue'] - row['Full8']) / row['Rescue']
        out.append(row)
    return out

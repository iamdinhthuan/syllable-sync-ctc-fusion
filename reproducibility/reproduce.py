"""Regenerate the reported numbers from the saved per-utterance records.

    python reproduce.py            # print all tables, write out/results.json
    python reproduce.py --check    # also compare with the values printed in the article

Only per-utterance edit counts and reference lengths are needed, so the script runs for every corpus
even where reference text is not redistributed.
"""
import json
import sys
from pathlib import Path

from metrics import ROOT, by_length, cer, load, paired_numpy, paired_python, wer

CORPORA = ['ViVOS', 'LSVSC', 'ViMD']
SUBSETS = ['I', 'R', 'T', 'IR', 'IRT']


def main_results():
    out = {}
    for c in CORPORA + ['FLEURS']:
        lm, resc, full = load(c, 'main_lm'), load(c, 'main_rescore'), load(c, 'main_full8')
        row = dict(utterances=len(full), LM=wer(lm), Rescore=wer(resc), Full8=wer(full), Full8_CER=cer(full),
                   caps_LM=sum(r['cap'] for r in lm.values()), caps_Full8=sum(r['cap'] for r in full.values()))
        if c != 'FLEURS':
            row['Greedy'] = wer(load(c, 'main_greedy'))
            row['Rescue'] = wer(load(c, 'main_rescue'))
            covered = [u for u, r in resc.items() if not r['empty_pool']]
            row['covered_utterances'] = len(covered)
            row['Rescore_minus_Full8_covered'] = paired_numpy(resc, full, covered)
            greedy = load(c, 'main_greedy')
            row['LM_minus_Full8_covered'] = paired_numpy(lm, full, covered)
            row['LM_minus_Full8_all'] = paired_numpy(lm, full)
            row['Rescore_minus_Full8_all'] = paired_numpy(resc, full)
            row['LM_with_rescue'] = 100 * sum((greedy[u]['e'] if resc[u]['empty_pool'] else lm[u]['e']) for u in lm) / sum(
                r['n'] for r in lm.values())
            row['covered'] = dict(LM=wer(lm, covered), Rescore=wer(resc, covered), Full8=wer(full, covered),
                                  tokens=sum(full[u]['n'] for u in covered))
            row['LM_without_capped'] = wer(lm, [u for u, r in lm.items() if not r['cap']])
            row['relative_reduction_vs_Rescore'] = 100 * (row['Rescore'] - row['Full8']) / row['Rescore']
            row['relative_reduction_vs_Greedy'] = 100 * (row['Greedy'] - row['Full8']) / row['Greedy']
        out[c] = row
    seen = set(json.loads((ROOT / 'data/ViVOS/train_text_recurrence_uids.json').read_text()))
    full = load('ViVOS', 'main_full8')
    out['ViVOS']['Full8_nonrecurring'] = dict(utterances=len(full) - len(seen),
                                              WER=wer(full, [u for u in full if u not in seen]))
    out['ViVOS']['recurrence'] = {
        part: {name: wer(load('ViVOS', f'main_{sysname}'), ids) for name, sysname in
               [('LM', 'lm'), ('Rescore', 'rescore'), ('Full8', 'full8')]}
        for part, ids in [('seen', [u for u in full if u in seen]), ('unseen', [u for u in full if u not in seen])]}
    return out


def length_profiles():
    out = {}
    for c in ['LSVSC', 'ViMD']:
        out[c] = by_length(dict(LM=load(c, 'main_lm'), Rescue=load(c, 'main_rescue'),
                                Bank32=load(c, 'main_bank32'), Full8=load(c, 'main_full8')))
    return out


def component_retune():
    settings = json.loads((ROOT / 'data/component_retune_settings.json').read_text())
    out = {}
    for c in CORPORA:
        irt = load(c, 'component_retuned_IRT')
        out[c] = {}
        for s in SUBSETS:
            rows = load(c, f'component_retuned_{s}')
            st = settings[f'{c}/{s}']
            rec = dict(setting=st['selected'], dev_WER=st['selected_dev']['WER'], test_WER=wer(rows), test_CER=cer(rows),
                       fixed_setting_test_WER=wer(load(c, f'component_fixed_{s}')), dev_points=st['n_points'],
                       bracketed=st['bracketed'])
            if s != 'IRT':
                rec['minus_IRT'] = paired_python(rows, irt)
            out[c][s] = rec
    return out


def pool_oracles():
    out = {}
    for c in CORPORA:
        out[c] = {s: wer(load(c, f'component_fixed_{s}'), edits='oe') for s in SUBSETS}
        out[c]['selected_IRT'] = wer(load(c, 'component_fixed_IRT'))
    return out


def bank32():
    out = {}
    for c in CORPORA + ['FLEURS']:
        bank, full = load(c, 'main_bank32'), load(c, 'main_full8')
        covered = [u for u, r in bank.items() if r['oe'] is not None]
        out[c] = dict(WER=wer(bank), CER=cer(bank), covered=len(covered), oracle_covered=wer(bank, covered, 'oe'),
                      Full8_covered=wer(full, covered))
        if c != 'FLEURS':
            out[c]['Bank32_minus_Full8'] = paired_numpy(bank, full)
    return out


def training_trials():
    """Lowest development CER (greedy decoding) and its epoch for each logged training run (Section S9)."""
    trials = json.loads((ROOT / 'data/training_trials.json').read_text())
    out = {}
    for name, t in trials.items():
        v = t['dev_CER']
        b = min(range(len(v)), key=v.__getitem__)
        out[name] = dict(corpus=t['corpus'], group=t['group'], best_dev_CER=v[b], best_epoch=b + 1, epochs_run=len(v))
    # length-matched view of the schedule comparison: the single schedule within the epochs the staged runs used
    staged = sum(len(trials[n]['dev_CER']) for n in ['x_lsvsc', 'x_lsvsc2'])  # 12 + 10 epochs
    out['single_schedule_within_staged_epochs'] = {'fl_lsvsc_s0': min(trials['fl_lsvsc_s0']['dev_CER'][:staged])}
    return out


def show(results):
    m = results['main_results']
    print('Main results (WER %)\n corpus      greedy     LM  rescore  +rescue  Full8  Full8-CER  caps')
    for c, r in m.items():
        g = f"{r['Greedy']:7.2f}" if 'Greedy' in r else '     --'
        q = f"{r['Rescue']:7.2f}" if 'Rescue' in r else '     --'
        print(f" {c:8s} {g} {r['LM']:7.2f} {r['Rescore']:7.2f}  {q} {r['Full8']:6.2f} {r['Full8_CER']:9.2f}   "
              f"{r['caps_LM']}/{r['caps_Full8']}")
    print('\nRescore - Full8 where rescoring has a finite completed candidate (points, 95% interval)')
    for c in CORPORA:
        d = m[c]['Rescore_minus_Full8_covered']
        print(f" {c:6s} {d['difference']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]  ({m[c]['covered_utterances']} utterances)")
    print('\nComponent retune (test WER %, subset - IRT with 95% interval)')
    for c, rows in results['component_retune'].items():
        for s, r in rows.items():
            d = r.get('minus_IRT')
            tail = '' if d is None else f"  {d['difference']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
            print(f" {c:6s} {s:3s} {r['setting']:14s} dev {r['dev_WER']:6.2f}  test {r['test_WER']:6.2f}{tail}")
    print('\nPool oracle WER % at the frozen Full8 settings')
    for c, r in results['pool_oracles'].items():
        print(f" {c:6s} " + '  '.join(f"{k} {v:.2f}" for k, v in r.items()))
    print('\nWER by reference length')
    for c, rows in results['length_profiles'].items():
        for r in rows:
            print(f" {c:6s} {r['bin']:6s} n={r['utterances']:5d}  LM {r['LM']:6.2f}  rescue {r['Rescue']:6.2f}  "
                  f"Bank32 {r['Bank32']:6.2f}  Full8 {r['Full8']:6.2f}")
    print('\nTraining trials (development CER %, greedy decoding; best epoch / epochs run)')
    for n, r in results['training_trials'].items():
        if n == 'single_schedule_within_staged_epochs':
            print(' single schedule within the staged epoch counts:', {k: round(v, 2) for k, v in r.items()})
            continue
        print(f" {n:15s} {r['corpus']:6s} {r['group']:16s} {r['best_dev_CER']:6.2f}  {r['best_epoch']:3d}/{r['epochs_run']}")
    print('\nBank32 (WER %, CER %, covered utterances, pool oracle, Full8 on covered)')
    for c, r in results['bank32'].items():
        print(f" {c:6s} {r['WER']:6.2f} {r['CER']:6.2f} {r['covered']:5d} {r['oracle_covered']:6.2f} {r['Full8_covered']:6.2f}")


def check(results):
    """Compare with the values printed in the article, at exactly the printed number of decimals."""
    expected = json.loads((ROOT / 'expected_values.json').read_text())
    bad, count = [], [0]

    def walk(exp, got, path):
        if isinstance(exp, dict):
            for k, v in exp.items():
                walk(v, got[k], path + [k])
        elif isinstance(exp, list):
            for i, v in enumerate(exp):
                walk(v, got[i], path + [str(i)])
        else:
            count[0] += 1
            if isinstance(exp, str):
                digits = len(exp.split('.')[1]) if '.' in exp else 0
                shown = f'{got:.{digits}f}'
                if shown == '-' + '0' * 1 + ('.' + '0' * digits if digits else ''):
                    shown = shown[1:]
                ok = shown == exp
            else:
                ok = got == exp
            if not ok:
                bad.append(('/'.join(path), exp, got))

    walk(expected, results, [])
    for item in bad:
        print('MISMATCH', *item)
    print(f"\nCHECK {'PASSED' if not bad else 'FAILED'}: {count[0]} values compared with the article at printed precision")
    return not bad


if __name__ == '__main__':
    results = dict(main_results=main_results(), component_retune=component_retune(), pool_oracles=pool_oracles(),
                   length_profiles=length_profiles(), bank32=bank32(), training_trials=training_trials())
    (ROOT / 'out').mkdir(exist_ok=True)
    (ROOT / 'out/results.json').write_text(json.dumps(results, indent=1))
    show(results)
    if '--check' in sys.argv:
        sys.exit(0 if check(results) else 1)

"""Select and print the LSVSC output cases by the article's fixed extreme-case rule.

Among test utterances with 12-30 reference tokens and no digit characters, cases are ranked by the
token-edit difference between the retuned LM output and Full8 (ties broken by utterance ID).
Rank 1 of each direction is the output figure of the main article; ranks 2-3 are in the supplement.

    python show_examples.py            # plain text (LSVSC)
    python show_examples.py --corpus FLEURS
    python show_examples.py --latex    # out/fig_outputs_rows.tex and out/examples_rows.tex

Marks: text recovered by the proposed decoder (correct in Full8, not matched in the LM output),
output tokens that differ from the reference, and bracketed notes for omitted reference tokens. Output strings are taken
verbatim from the saved records; only the bracketed notes and the English glosses are annotations.
All marks are relative to the corpus reference, which may itself contain transcription errors.
"""
import difflib
import sys

from metrics import ROOT, eligible, load

GLOSS = {  # authors' literal English glosses of the two references shown in the main-article figure
    'te01003_0100334': "and only one book I choose today, [I] would like to introduce a short-story collection that "
                       "once won the annual award of the Vietnam Writers' Association",
    'te00002_0000200': "the implementation of the socio-economic development plan and the state budget in the first "
                       "months of the year two thousand and twenty-two",
}
# Selected by the rule but not printed or distributed as audio or text: the transcript states a named
# individual's date and place of birth. Only its counts are in the records.
NOT_DISPLAYED = {'te01000_0100034'}


def opcodes(ref, hyp):
    return difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes()


def matched(ref, hyp):
    """Indices of reference tokens matched exactly by an output."""
    out = set()
    for tag, i1, i2, j1, j2 in opcodes(ref, hyp):
        if tag == 'equal':
            out.update(range(i1, i2))
    return out


def annotate(ref, hyp, other, proposed):
    """List of (kind, text) pieces: plain, rec (recovered by the proposed decoder), err, omit."""
    r, h = ref.split(), hyp.split()
    other_ok = matched(r, other.split())
    pieces = []

    def omit(tokens, name=True):
        if tokens:
            if name and len(tokens) <= 8:
                note = 'omitted: ' + ' '.join(tokens)
            else:
                note = f"{len(tokens)} {'more ' if not name else ''}token{'s' if len(tokens) > 1 else ''} omitted"
            pieces.append(('omit', note))

    for tag, i1, i2, j1, j2 in opcodes(r, h):
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                pieces.append(('rec' if proposed and i not in other_ok else 'plain', h[j]))
        elif tag == 'delete':
            omit(r[i1:i2])
        else:
            pieces.append(('err', ' '.join(h[j1:j2])))
            if tag == 'replace' and (i2 - i1) > (j2 - j1):
                omit(r[i1 + (j2 - j1):i2], name=False)  # which tokens of a substituted span are missing is ambiguous
    merged = []
    for kind, text in pieces:
        if merged and merged[-1][0] == kind and kind in ('plain', 'rec', 'err'):
            merged[-1] = (kind, merged[-1][1] + ' ' + text)
        else:
            merged.append((kind, text))
    return merged


def render(pieces, latex):
    if latex:
        macro = {'plain': '{}', 'rec': r'\rec{{{}}}', 'err': r'\err{{{}}}', 'omit': r'\omi{{{}}}'}
        return ' '.join(macro[k].format(t) for k, t in pieces)
    mark = {'plain': '{}', 'rec': '<<{}>>', 'err': '[{}]', 'omit': '({})'}
    return ' '.join(mark[k].format(t) for k, t in pieces)


def case_rows(u, lm, full, latex):
    ref = full[u]['ref']
    return [('Reference', ref, ''),
            ('Retuned LM', render(annotate(ref, lm[u]['hyp'], full[u]['hyp'], False), latex), lm[u]['e']),
            ('Full8 (proposed)', render(annotate(ref, full[u]['hyp'], lm[u]['hyp'], True), latex), full[u]['e'])]


if __name__ == '__main__':
    latex = '--latex' in sys.argv
    corpus = sys.argv[sys.argv.index('--corpus') + 1] if '--corpus' in sys.argv else 'LSVSC'
    lm, full = load(corpus, 'main_lm'), load(corpus, 'main_full8')
    cases = sorted(((lm[u]['e'] - full[u]['e'], u) for u, r in full.items() if eligible(r)), key=lambda x: (-x[0], x[1]))
    groups = [('Largest improvements', cases[:3]), ('Largest regressions', sorted(cases, key=lambda x: (x[0], x[1]))[:3])]
    if not latex:
        print('Marks: <<recovered by the proposed decoder>>  [differs from the reference]  (omitted reference tokens)')
        for title, group in groups:
            for rank, (delta, u) in enumerate(group, 1):
                print(f"\n{title}, rank {rank}: {u} (LM {lm[u]['e']} -> Full8 {full[u]['e']} token edits)")
                if u in NOT_DISPLAYED:
                    print('  (text not displayed: the transcript contains personal details of a named individual)')
                    continue
                for name, text, edits in case_rows(u, lm, full, False):
                    print(f'  {name:17s} {text}')
                if u in GLOSS:
                    print(f"  {'English gloss':17s} {GLOSS[u]}")
        sys.exit(0)
    if corpus != 'LSVSC':
        sys.exit('--latex writes the article rows and is defined for LSVSC only')
    out = ROOT / 'out'
    out.mkdir(exist_ok=True)
    main, supp = [], []
    for (title, group), panel, label in zip(groups, 'ab', ['Coverage recovered', 'Truncation failure']):
        for rank, (delta, u) in enumerate(group, 1):
            if u in NOT_DISPLAYED:
                continue
            uid = u.replace('_', r'\_')
            rows = case_rows(u, lm, full, True)
            if rank == 1:
                main.append(rf"\multicolumn{{2}}{{@{{}}l}}{{\textbf{{({panel}) {label}: {lm[u]['e']} $\rightarrow$ {full[u]['e']} token edits}} (LSVSC \texttt{{{uid}}})}}\\")
                main.append(rf"{rows[0][0]} & {rows[0][1]}\\")
                main.append(rf"English gloss & \textit{{{GLOSS[u]}}}\\")
                main += [rf"{name} & {text}\\" for name, text, _ in rows[1:]]
                main.append(r'\addlinespace')
            else:
                supp.append(rf"\multicolumn{{3}}{{l}}{{\textbf{{{title}, rank {rank}}} (\texttt{{{uid}}}; LM {lm[u]['e']} $\rightarrow$ Full8 {full[u]['e']} token edits)}}\\")
                supp += [rf"{name} & {text} & {edits}\\" for name, text, edits in rows]
                supp.append(r'\addlinespace')
    (out / 'fig_outputs_rows.tex').write_text('\n'.join(main[:-1]) + '\n')
    (out / 'examples_rows.tex').write_text('\n'.join(supp[:-1]) + '\n')
    print('wrote out/fig_outputs_rows.tex and out/examples_rows.tex')

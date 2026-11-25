"""Regenerate the data figures from the saved per-utterance records (Matplotlib).

    python make_figures.py        # writes out/*.pdf

Main-article Fig. 2 (WER by reference length) and Fig. 4 (component subsets), and the supplementary
retune-curve and Bank32 length figures. Fig. 1 and the output panels are hand-assembled artwork; the
cases shown in the output panels are selected by show_examples.py.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from metrics import ROOT, by_length, load, wer

OUT = ROOT / 'out'
OUT.mkdir(exist_ok=True)
INK = '#182D3B'
plt.rcParams.update({'font.family': ['Liberation Sans', 'DejaVu Sans'], 'font.size': 8.5, 'axes.linewidth': 0.6,
                     'pdf.fonttype': 42, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
                     'ytick.color': INK, 'axes.edgecolor': INK})
BLUE, ORANGE, PURPLE, GREEN, SLATE = '#0B6BB8', '#C8641E', '#6A3D9A', '#008B69', '#5E6B78'
SUBSETS = ['I', 'R', 'T', 'IR', 'IRT']


def length_figure(name, series, compact=False):
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.05) if compact else (7.1, 2.15), constrained_layout=True)
    for ax, (corpus, tag) in zip(axes, [('LSVSC', '(a) LSVSC'), ('ViMD', '(b) ViMD')]):
        rows = by_length({key: load(corpus, f'main_{sysname}') for key, sysname, *_ in series})
        x = range(len(rows))
        top = 20
        for key, _, label, color, marker, ls in series:
            y = [r[key] for r in rows]
            if max(y) > 20:
                top = max(top, max(y) + 2)
            ax.plot(x, y, ls=ls, marker=marker, ms=3 if compact else 4, lw=1.2 if compact else 1.4, color=color,
                    label=label)
            if not compact:
                ax.annotate(f'{y[-1]:.2f}', (len(rows) - 1, y[-1]), xytext=(6, 0), textcoords='offset points',
                            va='center', fontsize=7.5, color=color, fontweight='bold')
        ax.set_xticks(list(x))
        ax.set_xticklabels([r['bin'].replace('>=', '≥').replace('-', '–') for r in rows],
                           fontsize=6.5 if compact else 8)
        ax.set_ylim(0, top)
        ax.set_xlim(-0.3, len(rows) - (0.7 if compact else 0.4))
        if not compact or corpus == 'LSVSC':
            ax.set_ylabel('WER (%)')
        ax.set_xlabel('Reference length (tokens)', fontsize=7.5 if compact else None)
        if compact:
            ax.tick_params(axis='y', labelsize=7)
        ax.set_title(tag, loc='left', fontweight='bold', fontsize=9)
        ax.grid(axis='y', lw=0.4, color='#DDDDDD')
        ax.spines[['top', 'right']].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    if compact:
        fig.legend(h, l, loc='upper center', ncol=2, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, 1.2))
    else:
        fig.legend(h, l, loc='upper center', ncol=len(series), frameon=False, fontsize=7.8,
                   bbox_to_anchor=(0.5, 1.12))
    fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)


def component_figure():
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1), constrained_layout=True)
    for ax, (c, tag) in zip(axes, [('ViVOS', '(a) ViVOS'), ('LSVSC', '(b) LSVSC'), ('ViMD', '(c) ViMD')]):
        y = [wer(load(c, f'component_retuned_{s}')) for s in SUBSETS]
        fixed = [wer(load(c, f'component_fixed_{s}')) for s in SUBSETS]
        ax.bar(range(5), y, color=[SLATE] * 4 + [BLUE], width=0.68, zorder=2)
        ax.scatter(range(5), fixed, marker='o', facecolors='none', edgecolors=INK, s=18, lw=0.9, zorder=3,
                   label='IRT fusion settings (not retuned)')
        for i, v in enumerate(y):
            ax.text(i, 0.35, f'{v:.2f}', ha='center', va='bottom', fontsize=7.2, rotation=90, color='white',
                    fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels(SUBSETS)
        ax.set_ylim(0, 13.5)
        ax.set_ylabel('Test WER (%)')
        ax.set_title(tag, loc='left', fontweight='bold', fontsize=9)
        ax.grid(axis='y', lw=0.4, color='#DDDDDD', zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    h = [Patch(color=SLATE), Patch(color=BLUE)] + h
    l = ['Subset, common development retune', 'IRT, common development retune'] + l
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, 1.12))
    fig.savefig(OUT / 'fig4_component_retuned.pdf', bbox_inches='tight')
    plt.close(fig)


def retune_curves():
    """Development WER along the CTC-weight sweep at the original (alpha, beta), and the selected settings."""
    settings = json.loads((ROOT / 'data/component_retune_settings.json').read_text())
    style = {'I': (ORANGE, 's', '--'), 'R': (PURPLE, '^', ':'), 'T': (GREEN, 'D', '-.'), 'IR': (SLATE, 'v', '--'),
             'IRT': (BLUE, 'o', '-')}
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.0), constrained_layout=True)
    for ax, (c, tag) in zip(axes, [('ViVOS', '(a) ViVOS'), ('LSVSC', '(b) LSVSC'), ('ViMD', '(c) ViMD')]):
        low = 100
        for s in SUBSETS:
            st = settings[f'{c}/{s}']
            _, a0, b0 = map(float, st['original']['setting'].split(':'))
            pts = sorted((float(k.split(':')[0]), v['WER']) for k, v in st['dev_points'].items()
                         if abs(float(k.split(':')[1]) - a0) < 1e-9 and abs(float(k.split(':')[2]) - b0) < 1e-9)
            color, marker, ls = style[s]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, marker=marker, ms=3.2, lw=1.2 if s != 'IRT' else 1.8,
                    color=color, label=s)
            lam = float(st['selected'].split(':')[0])
            ax.scatter([lam], [st['selected_dev']['WER']], marker='*', s=70, color=color, edgecolors=INK, lw=0.5, zorder=4)
            low = min(low, st['selected_dev']['WER'])
        ax.set_ylim(low - 0.25, low + 3.2)
        ax.set_xlim(0, 1)
        ax.set_xlabel('CTC weight λ')
        ax.set_ylabel('Development WER (%)')
        ax.set_title(tag, loc='left', fontweight='bold', fontsize=9)
        ax.grid(axis='y', lw=0.4, color='#DDDDDD')
        ax.spines[['top', 'right']].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, 1.14))
    fig.savefig(OUT / 'figS_retune_dev_curves.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    base = [('LM', 'lm', 'Retuned LM proposals', ORANGE, 's', '--'),
            ('Rescue', 'rescue', 'Completed-score rescoring (+ rescue)', PURPLE, '^', ':'),
            ('Full8', 'full8', 'Online component fusion (Full8)', BLUE, 'o', '-')]
    length_figure('fig2_length_retuned.pdf', base)
    length_figure('figS_length_bank32.pdf',
                  [('LM', 'lm', 'Retuned LM', ORANGE, 's', '--'), ('Rescue', 'rescue', 'Rescoring (+ rescue)', PURPLE, '^', ':'),
                   ('Bank32', 'bank32', 'Bank32 rescoring', GREEN, 'D', '-.'), ('Full8', 'full8', 'Full8', BLUE, 'o', '-')],
                  compact=True)
    component_figure()
    retune_curves()
    print('wrote', sorted(p.name for p in OUT.glob('*.pdf')))

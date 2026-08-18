#!/usr/bin/env python3
"""Figures for the SPL letter.

Fig. 1 -- Distribution of P(no Condorcet winner) across configurations vs. polynomial degree s
          (30 configurations per M, two-seed average), symlog on y-axis.
Fig. 2 -- (a) Fraction of configurations exceeding 1% vs. s, compared with fraction
              of intransitive tournaments (hatched);
          (b) Dependence on the sample size N per decision.

Sources: verification/results_v3_gap_vs_s_M.json, *_repro.json, results_v10_N.json
"""
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# IEEE PDF eXpress rejects Type 3 fonts; 42 selects TrueType.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import os
ROOT = os.path.dirname(os.path.abspath(__file__)) + '/'
d = json.load(open(ROOT + 'verification/results_v3_gap_vs_s_M.json'))
r = json.load(open(ROOT + 'verification/results_v3_gap_vs_s_M_repro.json'))
n10 = json.load(open(ROOT + 'verification/results_v10_N.json'))

S = [1, 2, 3, 4, 5, 6]
MS = ['3', '4', '5', '6']
NS = [1, 2, 4, 8, 16, 32]

gap, cyc = {}, {}
for M in MS:
    a, b = d['random'][M], r['rerun'][M]
    gap[M] = np.array([[0.5 * (a[i]['by_s'][str(s)]['gap'] + b[i][str(s)]['gap'])
                        for s in S] for i in range(len(a))])
    cyc[M] = np.array([[0.5 * (a[i]['by_s'][str(s)]['cycle'] + b[i][str(s)]['cycle'])
                        for s in S] for i in range(len(a))])

plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7})
colors = {'3': '#1f77b4', '4': '#ff7f0e', '5': '#2ca02c', '6': '#d62728'}

# ------------------------------------------------------------------ Fig. 1
fig, ax = plt.subplots(figsize=(3.5, 2.15), dpi=300)
off = {'3': -0.27, '4': -0.09, '5': 0.09, '6': 0.27}
rng = np.random.default_rng(0)
lin = 1e-4
for M in MS:
    g = gap[M]
    for j, s in enumerate(S):
        x = s + off[M] + rng.uniform(-0.05, 0.05, len(g))
        ax.scatter(x, np.maximum(g[:, j], 0), s=3.5, color=colors[M], alpha=0.6,
                   lw=0, rasterized=True, label=f'$M={M}$' if j == 1 else None)
        ax.plot([s + off[M] - 0.09, s + off[M] + 0.09],
                [np.median(g[:, j])] * 2, color='k', lw=0.9)
ax.set_yscale('symlog', linthresh=lin, linscale=0.4)
ax.set_ylim(-lin * 0.45, 0.45)
ax.set_yticks([0, 1e-4, 1e-3, 1e-2, 1e-1])
ax.set_yticklabels(['0', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
ax.axhline(0.01, color='0.6', lw=0.5, ls=':')
ax.set_xticks(S)
ax.set_xlim(0.6, 6.5)
ax.set_xlabel(r'polynomial degree $s$')
ax.set_ylabel(r'$\mathrm{P}(\mathrm{no\ Condorcet\ winner})$')
ax.text(1.0, 3.5e-4, 'exactly 0', ha='center', va='bottom', fontsize=6, color='0.35')
ax.legend(frameon=False, fontsize=6.5, loc='upper right', ncol=4, columnspacing=0.7,
          handletextpad=0.15, borderaxespad=0.3, markerscale=1.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout(pad=0.25)
fig.savefig(ROOT + 'figures/fig1_gap_vs_s_M.pdf')

# ------------------------------------------------------------------ Fig. 2
# s is discrete -> bar plot rather than lines. Since {P(gap) > 1%} is a subset
# of {P(cycle) > 1%} (as P(cycle) >= P(gap) pointwise per configuration),
# the bar decomposes: solid bottom = undecided events, hatched top = configurations
# where the tournament is intransitive but a Condorcet winner still exists.
fig, ax1 = plt.subplots(figsize=(3.4, 2.05), dpi=300)
w = 0.2
for t, M in enumerate(MS):
    fg = (gap[M] > 0.01).mean(0)
    fc = (cyc[M] > 0.01).mean(0)
    xs = np.arange(len(S)) + (t - 1.5) * w
    ax1.bar(xs, fg, width=w * 0.92, color=colors[M], lw=0,
            label=f'$M={M}$')
    ax1.bar(xs, np.maximum(fc - fg, 0), width=w * 0.92, bottom=fg,
            color=colors[M], alpha=0.28, lw=0.4, edgecolor=colors[M], hatch='///')
ax1.set_xticks(np.arange(len(S)))
ax1.set_xticklabels([str(v) for v in S])
ax1.set_ylim(0, 1.16)
ax1.set_xlabel(r'polynomial degree $s$')
ax1.set_ylabel(r'configs. with $\mathrm{P}>1\%$')
ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
h, l = ax1.get_legend_handles_labels()
ax1.legend(h, l, frameon=False, fontsize=6, loc='upper left', ncol=4,
           columnspacing=0.7, handlelength=1.0, handletextpad=0.3,
           borderaxespad=0.15)
ax1.text(0.015, 0.855, 'solid: no Condorcet winner;  hatched: intransitive only',
         transform=ax1.transAxes, fontsize=5.5, color='0.3')
ax1.text(0, 0.022, 'none', ha='center', va='bottom', fontsize=5.5, color='0.45')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
fig.tight_layout(pad=0.25)
fig.savefig(ROOT + 'figures/fig2_frac.pdf')

# ------------------------------------------------------------------ Fig. 3
fig, ax2 = plt.subplots(figsize=(3.4, 1.62), dpi=300)
arr = np.array([[row[str(k)] for k in NS] for row in n10['sweep']])
for row in arr:
    ax2.plot(NS, np.maximum(row, 1e-5), '-', color='0.78', lw=0.4, alpha=0.8)
ax2.plot(NS, np.median(arr, 0), '-o', color='#d62728', ms=3, lw=1.3, label='median')
ax2.plot(NS, [max(n10['worst'][str(k)], 1e-5) for k in NS], '-s', color='k',
         ms=2.5, lw=1.0, label='worst configuration')
ax2.set_xscale('log', base=2)
ax2.set_yscale('log')
ax2.set_xticks(NS)
ax2.set_xticklabels([str(k) for k in NS])
ax2.set_ylim(1e-4, 0.3)
ax2.set_xlabel(r'observations per decision $N$')
ax2.set_ylabel(r'$\mathrm{P}$ (no winner)')
ax2.axhline(0.01, color='0.6', lw=0.5, ls=':')
ax2.legend(frameon=False, fontsize=6, loc='lower left', handlelength=1.3,
           borderaxespad=0.2, labelspacing=0.15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
fig.tight_layout(pad=0.25)
fig.savefig(ROOT + 'figures/fig3_N.pdf')
print('saved fig1, fig2, fig3')

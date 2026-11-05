#!/usr/bin/env python3
"""读取三个 seed 的预测，绘制分布和均值，只保存 PNG 图片。"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs'
OUT = ROOT / 'outputs/prediction_distributions'
COLORS = {'source': '#0072B2', 'zero_shot': '#D55E00', 'head_fine_tuned': '#0072B2'}
SEEDS = {13: 'o', 14: 's', 15: '^'}
WIDTH = 0.080  # 分布的最大横向半宽，不代表真实 R_M 的变化。

SOURCE = [
    'Holmboe:Rib020_a05_Re2000',
    'KH:Ri020_a05_Re1000',
    'Holmboe:Rib016_a10_Re2000',
    'KH:Ri016_a10_Re2000',
]

TARGET = [
    'derechoHB_Pr4:asymHol_Ri030_a05_Re1000',
    'derechoHB_Pr4:asymKH_Ri016_a05_Re1000',
    'derechoPr7_Holmboe:Ri016_a10_single',
    'derechoPr7_Holmboe:Ri030_a05_single',
    'noisyKH:Re2000_Ri016_Pr1_A01',
    'Alexis:asym_KH',
]

# case 前缀: 图例、轮廓线型。
DATASETS = {
    'derechoHB_Pr4': ('$Pr=4$', 'solid'),
    'derechoPr7_Holmboe': ('$Pr=7$', (0, (4, 2))),
    'noisyKH': ('Noisy KH', (0, (1, 2))),
    'Alexis': ('$Pr=9$', (0, (5, 2, 1, 2))),
}


def load_predictions():
    groups, per_seed, truth = defaultdict(list), defaultdict(list), {}
    with (INPUT / 'plotted_panel_predictions.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            condition, case, seed = row['condition'], row['case_id'], int(row['seed'])
            value = float(row['prediction'])
            truth[case] = float(row['true_rm'])
            groups[condition, case].append(value)
            per_seed[condition, case, seed].append(value)
    means = {key: np.mean(values) for key, values in per_seed.items()}
    return groups, truth, means


def make_figure():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.labelsize': 13,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.7))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.18, wspace=0.25)
    handles = [
        Line2D([], [], marker=marker, linestyle='none', markersize=7.5,
               markerfacecolor='#404040', markeredgecolor='white', label=f'Seed {seed}: case mean')
        for seed, marker in SEEDS.items()
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.52, 0.935),
               ncol=3, frameon=False, fontsize=10)
    titles = ['(a) Source: $Pr=1$', '(b) Transfer: all target cases']
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc='left', fontsize=14, pad=10)
        ax.set(xlim=(0, 1.12), ylim=(0, 1.12), xlabel=r'True $R_M$', ylabel=r'Predicted $R_M$')
        ax.set_aspect('equal', adjustable='box')
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_ticks(np.arange(0, 1.01, 0.25))
            axis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.plot([0, 1.12], [0, 1.12], '--', color='.65', lw=1.2, zorder=0)
    states = [('zero_shot', 'Zero-shot'), ('head_fine_tuned', 'Head fine-tuned')]
    handles = [Line2D([], [], color=COLORS[key], lw=5, label=label) for key, label in states]
    legend = axes[1].legend(handles=handles, loc='upper left', frameon=False, fontsize=10)
    axes[1].add_artist(legend)
    handles = [Line2D([], [], color='.4', lw=1.1, linestyle=style, label=label)
               for label, style in DATASETS.values()]
    axes[1].legend(handles=handles, loc='lower right', ncol=2, frameon=False, fontsize=9.5)
    return fig, axes


def draw_case(ax, case, condition, x, values, means):
    """分布左右对称；panel 点和 seed 均值的横坐标都等于真值。"""
    color = COLORS[condition]
    _, style = DATASETS.get(case.split(':')[0], ('Source', 'solid'))
    values = np.asarray(values)
    y = np.linspace(values.min(), values.max(), 512)
    probability = gaussian_kde(values, bw_method='scott')(y)
    probability /= np.trapezoid(probability, y)
    width = WIDTH * probability / probability.max()
    ax.fill_betweenx(y, x - width, x + width, facecolor=color, alpha=0.16, linewidth=0, zorder=2)
    for edge in (x - width, x + width):
        ax.plot(edge, y, color=color, lw=1.05, alpha=0.8, linestyle=style, zorder=3)
    ax.scatter(np.full(len(values), x), values, color=color, s=1.1, alpha=0.055,
               linewidths=0, zorder=3, rasterized=True)
    for seed, marker in SEEDS.items():
        ax.scatter(x, means[condition, case, seed], marker=marker, s=43,
                   facecolors='white' if condition == 'zero_shot' else color,
                   edgecolors=color if condition == 'zero_shot' else 'white',
                   linewidths=0.95, zorder=5)


def main():
    groups, truth, means = load_predictions()
    fig, axes = make_figure()
    panels = [(axes[0], SOURCE, ['source']),
              (axes[1], TARGET, ['zero_shot', 'head_fine_tuned'])]
    for ax, cases, conditions in panels:
        for case in cases:
            for condition in conditions:
                draw_case(ax, case, condition, truth[case], groups[condition, case], means)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'prediction_distributions_centered.png'
    fig.savefig(path, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()

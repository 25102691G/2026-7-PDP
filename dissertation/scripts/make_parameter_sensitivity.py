"""Plot three slices of the saved PDP v2 development screen."""

import json
import os
from pathlib import Path
from statistics import mean, stdev

PAPER = Path(__file__).resolve().parents[1]
SOURCE = (
    PAPER.parent / 'docs/pdp_research/dataset_parameter_screen_20261112_analysis'
    / 'ALL_CELL_COMPARISONS.json'
)
os.environ['MPLCONFIGDIR'] = str(PAPER / 'tooling/mpl-cache')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    datasets = {
        'email': ('Email', '#21638c', 'o'),
        'bitcoin': ('Bitcoin', '#a54517', 's'),
        'lastfm': ('LastFM', '#337b4b', '^'),
        'hepph': ('HepPh', '#87518b', 'D'),
        'facebook': ('Facebook', '#444444', 'v'),
    }
    candidates = {
        'baseline', 'size_a', 'size_b', 'steps300', 'steps500',
        'frequency_a', 'frequency_b',
    }
    rows = [
        row for row in json.loads(SOURCE.read_text())
        if row['method'] == 'pdp_v2' and row['candidate'] in candidates
    ]
    assert {(row['dataset'], row['candidate']) for row in rows} == {
        (dataset, candidate) for dataset in datasets for candidate in candidates
    }
    assert len(rows) == len(datasets) * len(candidates)
    for row in rows:
        values = row['seeds']
        assert len(values) == row['n_runs'] == 3
        assert abs(mean(values) - row['mean']) < 1e-10
        assert abs(stdev(values) - row['sample_sd']) < 1e-10

    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 10,
        'pdf.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.15), sharey=True)
    panels = [
        ('n', {'M': 4, 'T': 100}, 1.1, '(a) Subgraph size', r'$n$'),
        ('T', {'n': 40, 'M': 4}, 10, '(b) Steps and noise', r'Updates $T$'),
        ('M', {'n': 40, 'T': 100}, 0.11, '(c) SCS threshold', r'$M$'),
    ]
    for ax, (varied, fixed, offset, title, xlabel) in zip(axes, panels):
        panel_rows = [
            row for row in rows
            if all(row[parameter] == value for parameter, value in fixed.items())
        ]
        for index, (dataset, (label, color, marker)) in enumerate(datasets.items()):
            series = sorted(
                (row for row in panel_rows if row['dataset'] == dataset),
                key=lambda row: row[varied],
            )
            assert len({row[varied] for row in series}) == len(series) == 3
            ax.errorbar(
                [row[varied] + (index - 2) * offset for row in series],
                [row['mean'] for row in series],
                yerr=[row['sample_sd'] for row in series],
                color=color, marker=marker, label=label, linestyle='-',
                linewidth=0.9, markersize=4, capsize=2, elinewidth=0.8,
            )
        ax.set_xticks(sorted({row[varied] for row in panel_rows}))
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.25)

    axes[0].set_ylabel('Coverage (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=0.8)
    fig.savefig(
        PAPER / 'figures/parameter_sensitivity.pdf',
        metadata={'CreationDate': None, 'ModDate': None},
    )
    plt.close(fig)
    print(f'Plotted {len(rows)} distinct PDP v2 configurations ({3 * len(rows)} saved runs).')


if __name__ == '__main__':
    main()

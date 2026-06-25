"""Generate benchmark charts from summary and raw timing CSVs.

Phase 2 usage (single runtime — pipeline breakdown + percentiles + timeline):
    python3 scripts/plot_results.py \
        --baselines results/pytorch_baseline.csv \
        --raw       results/pytorch_raw_timings.csv \
        --labels    pytorch \
        --output-dir plots/

Phase 6 usage (multi-runtime comparison):
    python3 scripts/plot_results.py \
        --baselines results/pytorch_baseline.csv \
                    results/onnx_benchmark.csv \
                    results/tensorrt_fp32.csv \
                    results/tensorrt_fp16.csv \
        --labels    pytorch onnx trt-fp32 trt-fp16 \
        --output-dir plots/

Charts produced
---------------
Always:
  pipeline_breakdown.png    — stacked bar of mean stage times per runtime
  latency_percentiles.png   — grouped bar of p50/p90/p99 per stage (single runtime)
                              OR per runtime total latency (multi-runtime)
  fps_comparison.png        — FPS bar chart per runtime

When --raw is provided (single runtime only):
  latency_timeline.png      — per-frame total latency over time
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for Colab / headless
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.metrics import fps_from_mean_ms

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi':        150,
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'axes.labelsize':    11,
    'axes.titlesize':    13,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
})

STAGE_COLORS = {
    'read':       '#8da0cb',
    'preprocess': '#66c2a5',
    'inference':  '#fc8d62',
    'postprocess':'#e78ac3',
}

RUNTIME_COLORS = {
    'pytorch':  '#4c72b0',
    'onnx':     '#55a868',
    'trt-fp32': '#c44e52',
    'trt-fp16': '#dd8452',
}

PIPELINE_STAGES = ['read', 'preprocess', 'inference', 'postprocess']


def _get_color(key: str, palette: dict) -> str:
    return palette.get(key, '#999999')


# ── Individual chart functions ────────────────────────────────────────────────

def plot_pipeline_breakdown(all_summaries: list[pd.DataFrame], labels: list[str], out_path: str):
    """Stacked horizontal bar: mean time per stage, one bar per runtime."""
    fig, ax = plt.subplots(figsize=(10, max(2.5, 1.2 * len(labels))))

    bar_h  = 0.5
    y_pos  = np.arange(len(labels))
    lefts  = np.zeros(len(labels))

    for stage in PIPELINE_STAGES:
        values = []
        for df in all_summaries:
            row = df[df['stage'] == stage]
            values.append(row['mean_ms'].iloc[0] if not row.empty else 0.0)
        values = np.array(values)
        ax.barh(y_pos, values, bar_h, left=lefts,
                color=_get_color(stage, STAGE_COLORS), label=stage)
        lefts += values

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Mean latency (ms)')
    ax.set_title('Pipeline Stage Breakdown — Mean Latency per Runtime')
    ax.legend(loc='lower right', ncol=len(PIPELINE_STAGES), fontsize=8)

    # Annotate total
    for i, (left, label) in enumerate(zip(lefts, labels)):
        ax.text(left + 0.3, i, f'{left:.1f} ms\n({fps_from_mean_ms(left):.1f} FPS)',
                va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'Saved → {out_path}')


def plot_latency_percentiles_single(summary: pd.DataFrame, label: str, out_path: str):
    """Grouped bar of p50/p90/p99 for each pipeline stage (single runtime)."""
    stages = [s for s in PIPELINE_STAGES if not summary[summary['stage'] == s].empty]
    x      = np.arange(len(stages))
    width  = 0.25

    p50 = [summary[summary['stage'] == s]['p50_ms'].iloc[0] for s in stages]
    p90 = [summary[summary['stage'] == s]['p90_ms'].iloc[0] for s in stages]
    p99 = [summary[summary['stage'] == s]['p99_ms'].iloc[0] for s in stages]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, p50, width, label='p50', color='#4c72b0', alpha=0.85)
    ax.bar(x,         p90, width, label='p90', color='#dd8452', alpha=0.85)
    ax.bar(x + width, p99, width, label='p99', color='#c44e52', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Latency Percentiles per Stage — {label}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'Saved → {out_path}')


def plot_latency_percentiles_multi(all_summaries: list[pd.DataFrame], labels: list[str], out_path: str):
    """Grouped bar of p50/p90/p99 on total latency, one group per runtime."""
    x     = np.arange(len(labels))
    width = 0.25

    def _total(df, col):
        row = df[df['stage'] == 'total']
        return row[col].iloc[0] if not row.empty else 0.0

    p50 = [_total(df, 'p50_ms') for df in all_summaries]
    p90 = [_total(df, 'p90_ms') for df in all_summaries]
    p99 = [_total(df, 'p99_ms') for df in all_summaries]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, p50, width, label='p50', color='#4c72b0', alpha=0.85)
    ax.bar(x,         p90, width, label='p90', color='#dd8452', alpha=0.85)
    ax.bar(x + width, p99, width, label='p99', color='#c44e52', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('End-to-end latency (ms)')
    ax.set_title('End-to-End Latency Percentiles by Runtime')
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'Saved → {out_path}')


def plot_fps_comparison(all_summaries: list[pd.DataFrame], labels: list[str], out_path: str):
    """Bar chart of FPS derived from mean total latency per runtime."""
    fps_values = []
    for df in all_summaries:
        row = df[df['stage'] == 'total']
        fps_values.append(fps_from_mean_ms(row['mean_ms'].iloc[0]) if not row.empty else 0.0)

    colors = [_get_color(l, RUNTIME_COLORS) for l in labels]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, fps_values, color=colors, alpha=0.85, width=0.5)

    for bar, val in zip(bars, fps_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('FPS (higher is better)')
    ax.set_title('End-to-End FPS by Runtime')
    ax.set_ylim(0, max(fps_values) * 1.2)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'Saved → {out_path}')


def plot_latency_timeline(raw_df: pd.DataFrame, label: str, out_path: str):
    """Per-frame total latency line chart to visualise variance."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw_df['frame'], raw_df['total_ms'], linewidth=0.8,
            color='#4c72b0', alpha=0.7, label='total_ms')
    ax.axhline(raw_df['total_ms'].mean(), color='#c44e52', linewidth=1.5,
               linestyle='--', label=f'mean = {raw_df["total_ms"].mean():.1f} ms')

    ax.set_xlabel('Frame index')
    ax.set_ylabel('Total latency (ms)')
    ax.set_title(f'Per-Frame Latency Timeline — {label}')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'Saved → {out_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Generate benchmark charts')
    p.add_argument('--baselines',   nargs='+', required=True,
                   help='One or more summary CSV files (from benchmark.py)')
    p.add_argument('--raw',         default=None,
                   help='Raw per-frame CSV for timeline chart (single runtime only)')
    p.add_argument('--labels',      nargs='+', default=None,
                   help='Runtime labels matching --baselines order')
    p.add_argument('--output-dir',  default='plots/',
                   help='Directory to write PNG files')
    return p.parse_args()


def main():
    args = parse_args()

    all_summaries = [pd.read_csv(f) for f in args.baselines]
    labels = args.labels if args.labels else [f'runtime_{i}' for i in range(len(args.baselines))]

    if len(labels) != len(all_summaries):
        print('[ERROR] --labels count must match --baselines count')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    out = lambda name: os.path.join(args.output_dir, name)

    is_single = len(all_summaries) == 1

    # Always generated
    plot_pipeline_breakdown(all_summaries, labels, out('pipeline_breakdown.png'))
    plot_fps_comparison(all_summaries,     labels, out('fps_comparison.png'))

    if is_single:
        plot_latency_percentiles_single(all_summaries[0], labels[0],
                                        out('latency_percentiles.png'))
    else:
        plot_latency_percentiles_multi(all_summaries, labels,
                                       out('latency_percentiles.png'))

    # Timeline — only when raw CSV is provided
    if args.raw:
        if not is_single:
            print('[WARNING] --raw is only used for single-runtime timeline, ignoring for multi-runtime.')
        else:
            raw_df = pd.read_csv(args.raw)
            plot_latency_timeline(raw_df, labels[0], out('latency_timeline.png'))

    print(f'\nAll charts written to {args.output_dir}')


if __name__ == '__main__':
    main()

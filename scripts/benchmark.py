"""Phase 2: Aggregate per-frame raw timings into a benchmark summary.

Reads the per-frame CSV produced by any inference script and outputs one
summary row per pipeline stage with mean / std / p50 / p90 / p99 / max.

Usage:
    python3 scripts/benchmark.py \
        --results results/pytorch_raw_timings.csv \
        --output  results/pytorch_baseline.csv \
        --runtime pytorch
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import compute_stats, fps_from_mean_ms

STAGE_COLS = [
    'preprocess_ms',
    'inference_ms',
    'postprocess_ms',
    'total_ms',
]


def parse_args():
    p = argparse.ArgumentParser(description='Phase 2: Compute benchmark statistics from raw per-frame timings')
    p.add_argument('--results', default='results/pytorch_raw_timings.csv',
                   help='Raw per-frame timing CSV (output of run_*_video.py)')
    p.add_argument('--output',  default='results/pytorch_baseline.csv',
                   help='Output summary CSV path')
    p.add_argument('--runtime', default='pytorch',
                   help='Runtime label written into the output (pytorch / compile / onnx-cuda / trt-fp32 / trt-fp16)')
    p.add_argument('--warmup-frames', type=int, default=20,
                   help='Number of leading frames to discard before computing stats (default: 20)')
    return p.parse_args()


def build_summary(df: pd.DataFrame, runtime: str) -> pd.DataFrame:
    rows = []
    for col in STAGE_COLS:
        if col not in df.columns:
            continue
        s = compute_stats(df[col].dropna().tolist())
        rows.append({
            'runtime': runtime,
            'stage':   col.replace('_ms', ''),
            'count':   s['count'],
            'mean_ms': round(s['mean_ms'], 3),
            'std_ms':  round(s['std_ms'],  3),
            'min_ms':  round(s['min_ms'],  3),
            'p50_ms':  round(s['p50_ms'],  3),
            'p90_ms':  round(s['p90_ms'],  3),
            'p99_ms':  round(s['p99_ms'],  3),
            'max_ms':  round(s['max_ms'],  3),
        })
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame):
    header = f'{"Stage":<16} {"Mean":>8} {"Std":>7} {"p50":>8} {"p90":>8} {"p99":>8} {"Max":>8}'
    print(header)
    print('─' * len(header))
    for _, row in summary.iterrows():
        print(
            f'{row["stage"]:<16} '
            f'{row["mean_ms"]:>8.2f} '
            f'{row["std_ms"]:>7.2f} '
            f'{row["p50_ms"]:>8.2f} '
            f'{row["p90_ms"]:>8.2f} '
            f'{row["p99_ms"]:>8.2f} '
            f'{row["max_ms"]:>8.2f}'
        )

    total_rows = summary[summary['stage'] == 'total']
    if not total_rows.empty:
        mean_total = total_rows.iloc[0]['mean_ms']
        p50_total  = total_rows.iloc[0]['p50_ms']
        print(f'\nEnd-to-end FPS (mean total): {fps_from_mean_ms(mean_total):.1f}')
        print(f'End-to-end FPS (p50  total): {fps_from_mean_ms(p50_total):.1f}')


def main():
    args = parse_args()

    df = pd.read_csv(args.results)
    print(f'Runtime        : {args.runtime}')
    print(f'Source         : {args.results}  ({len(df)} frames total)')

    if args.warmup_frames > 0:
        df = df.iloc[args.warmup_frames:].reset_index(drop=True)
        print(f'Warmup dropped : first {args.warmup_frames} frames excluded from stats')

    print(f'Frames used    : {len(df)}\n')

    summary = build_summary(df, args.runtime)
    print_summary(summary)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f'\nSaved summary → {args.output}')


if __name__ == '__main__':
    main()

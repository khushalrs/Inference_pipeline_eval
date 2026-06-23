"""Phase 5, Optimization 2: Confidence threshold sweep on TRT FP16 pipeline.

A higher confidence threshold pre-filters more raw boxes before NMS, reducing
postprocessing time but potentially missing low-confidence true positives.

This sweep quantifies:
  - how much postprocessing time changes with conf threshold
  - how detection count varies (quality / recall proxy)
  - the net FPS change on the fastest runtime (TRT FP16)

Usage:
    python3 scripts/optimize_conf_threshold.py \
        --video       data/clip.mp4 \
        --engine      models/yolo11n_fp16.engine \
        --conf-values 0.05 0.15 0.25 0.50 0.75 \
        --results     results/conf_sweep.csv \
        --plot        plots/conf_sweep.png
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.timing import CUDATimer
from src.video_io import VideoReader
from src.preprocessing import letterbox, to_numpy_input
from src.postprocessing import run_nms, scale_boxes
from src.metrics import compute_stats, fps_from_mean_ms
from src.trt_runner import TRTRunner


def parse_args():
    p = argparse.ArgumentParser(description='Phase 5: Confidence threshold sweep on TRT FP16')
    p.add_argument('--video',       default='data/clip.mp4')
    p.add_argument('--engine',      default='models/yolo11n_fp16.engine')
    p.add_argument('--conf-values', type=float, nargs='+',
                   default=[0.05, 0.15, 0.25, 0.50, 0.75],
                   help='Confidence thresholds to sweep')
    p.add_argument('--iou',         type=float, default=0.45)
    p.add_argument('--warmup',      type=int,   default=10)
    p.add_argument('--input-size',  type=int,   default=640)
    p.add_argument('--results',     default='results/conf_sweep.csv')
    p.add_argument('--plot',        default='plots/conf_sweep.png')
    return p.parse_args()


def run_one_conf(runner: TRTRunner, video_path: str,
                 conf: float, iou: float, input_size: int) -> dict:
    """Run full video with given conf threshold, return aggregated stats."""
    input_shape = (input_size, input_size)
    reader      = VideoReader(video_path)

    inf_ms, post_ms, total_ms_list = [], [], []
    det_counts = []

    with reader:
        while True:
            ret, frame = reader.read()
            if not ret:
                break

            img_lb, _, pad = letterbox(frame, input_shape)
            arr_np = to_numpy_input(img_lb)

            with CUDATimer() as t_inf:
                raw_t = runner.infer(arr_np)

            with CUDATimer() as t_post:
                dets = run_nms(raw_t, conf, iou)
                det  = dets[0]
                if len(det):
                    det[:, :4] = scale_boxes(
                        det[:, :4],
                        orig_shape=(reader.height, reader.width),
                        input_shape=input_shape,
                        pad=pad,
                    )

            total = t_inf.elapsed_ms + t_post.elapsed_ms
            inf_ms.append(t_inf.elapsed_ms)
            post_ms.append(t_post.elapsed_ms)
            total_ms_list.append(total)
            det_counts.append(len(det))

    ts = compute_stats(total_ms_list)
    return {
        'conf_threshold':  conf,
        'frames':          len(total_ms_list),
        'mean_inf_ms':     round(compute_stats(inf_ms)['mean_ms'],   3),
        'mean_post_ms':    round(compute_stats(post_ms)['mean_ms'],  3),
        'mean_total_ms':   round(ts['mean_ms'], 3),
        'p50_total_ms':    round(ts['p50_ms'],  3),
        'p99_total_ms':    round(ts['p99_ms'],  3),
        'fps':             round(fps_from_mean_ms(ts['mean_ms']), 1),
        'mean_detections': round(sum(det_counts) / max(len(det_counts), 1), 2),
    }


def plot_sweep(rows: list, plot_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.3,
    })

    labels    = [str(r['conf_threshold']) for r in rows]
    post_vals = [r['mean_post_ms']    for r in rows]
    fps_vals  = [r['fps']             for r in rows]
    det_vals  = [r['mean_detections'] for r in rows]

    # Highlight baseline (conf=0.25)
    colors = ['#c44e52' if abs(r['conf_threshold'] - 0.25) < 0.01 else '#4c72b0'
              for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Postprocessing time vs conf
    axes[0].bar(labels, post_vals, color=colors, alpha=0.85, width=0.5)
    axes[0].set_title('NMS Postprocessing Time\nvs Confidence Threshold')
    axes[0].set_xlabel('Confidence threshold')
    axes[0].set_ylabel('Mean postprocess time (ms)')

    # FPS vs conf
    bars = axes[1].bar(labels, fps_vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, fps_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    axes[1].set_title('End-to-End FPS\n(inf + postprocess)')
    axes[1].set_xlabel('Confidence threshold')
    axes[1].set_ylabel('FPS')
    axes[1].set_ylim(0, max(fps_vals) * 1.25)

    # Detections vs conf
    axes[2].plot(labels, det_vals, 'o-', color='#dd8452', linewidth=2, markersize=9)
    for x, y in zip(labels, det_vals):
        axes[2].annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=9)
    axes[2].set_title('Mean Detections per Frame\nvs Confidence Threshold')
    axes[2].set_xlabel('Confidence threshold')
    axes[2].set_ylabel('Detections')
    axes[2].set_ylim(0, max(det_vals) * 1.4 if det_vals else 1)

    # Red bar = baseline label
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor='#c44e52', alpha=0.85, label='baseline (0.25)'),
                  Patch(facecolor='#4c72b0', alpha=0.85, label='other')]
    axes[0].legend(handles=legend_els, fontsize=8)

    plt.suptitle('Confidence Threshold Sweep — TRT FP16', fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'Saved → {plot_path}')


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print('[ERROR] Conf sweep requires CUDA GPU.')
        sys.exit(1)

    if not os.path.exists(args.engine):
        print(f'[ERROR] Engine not found: {args.engine}')
        print('        Run Phase 4 (build_tensorrt_engine.py) first.')
        sys.exit(1)

    print(f'Loading TRT FP16 engine: {args.engine}')
    runner = TRTRunner(args.engine, device='cuda')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

    print(f'\nWarming up ({args.warmup} iters) ...')
    dummy_np = np.zeros((1, 3, args.input_size, args.input_size), dtype=np.float32)
    for _ in range(args.warmup):
        runner.infer(dummy_np)
    torch.cuda.synchronize()

    rows = []
    for conf in args.conf_values:
        print(f'\n── conf={conf} ───────────────────────────────────────')
        row = run_one_conf(runner, args.video, conf, args.iou, args.input_size)
        rows.append(row)
        print(f'  FPS={row["fps"]:.1f}  post={row["mean_post_ms"]:.2f}ms  '
              f'dets={row["mean_detections"]:.1f}')

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    with open(args.results, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Print summary table
    print(f'\n{"Conf":<6} {"FPS":>6} {"post_ms":>9} {"dets":>6}')
    print('─' * 30)
    for r in rows:
        marker = ' ← baseline' if abs(r['conf_threshold'] - 0.25) < 0.01 else ''
        print(f'{r["conf_threshold"]:<6} {r["fps"]:>6.1f} '
              f'{r["mean_post_ms"]:>9.2f} {r["mean_detections"]:>6.1f}{marker}')

    # Insight summary
    baseline = next((r for r in rows if abs(r['conf_threshold'] - 0.25) < 0.01), rows[0])
    best     = max(rows, key=lambda r: r['fps'])
    gain     = best['fps'] - baseline['fps']
    print(f'\nBaseline conf=0.25  → {baseline["fps"]:.1f} FPS, '
          f'{baseline["mean_post_ms"]:.2f}ms post, '
          f'{baseline["mean_detections"]:.1f} dets/frame')
    print(f'Best     conf={best["conf_threshold"]}    → {best["fps"]:.1f} FPS '
          f'(+{gain:.1f} FPS gain, {best["mean_detections"]:.1f} dets/frame)')
    print(f'\nSaved → {args.results}')

    plot_sweep(rows, args.plot)


if __name__ == '__main__':
    main()

"""Phase 5, Optimization 1: Resolution sweep on the PyTorch pipeline.

Tests multiple square input sizes to quantify the speed vs detection quality
tradeoff. No model rebuild needed — letterbox handles any resolution.

Metrics collected per resolution:
  - mean / p50 / p99 latency per stage
  - end-to-end FPS
  - mean detections per frame (proxy for recall)

Usage:
    python3 scripts/optimize_resolution.py \
        --video       data/clip.mp4 \
        --model       yolo11n.pt \
        --resolutions 320 480 640 \
        --results     results/resolution_sweep.csv \
        --plot        plots/resolution_sweep.png \
        --device      cuda
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.timing import CUDATimer
from src.video_io import VideoReader
from src.preprocessing import letterbox, to_tensor
from src.postprocessing import run_nms, scale_boxes
from src.metrics import compute_stats, fps_from_mean_ms


def parse_args():
    p = argparse.ArgumentParser(description='Phase 5: Resolution sweep')
    p.add_argument('--video',       default='data/clip.mp4')
    p.add_argument('--model',       default='yolo11n.pt')
    p.add_argument('--resolutions', type=int, nargs='+', default=[320, 480, 640],
                   help='Square input sizes to test (e.g. 320 480 640)')
    p.add_argument('--conf',        type=float, default=0.25)
    p.add_argument('--iou',         type=float, default=0.45)
    p.add_argument('--warmup',      type=int,   default=10)
    p.add_argument('--results',     default='results/resolution_sweep.csv')
    p.add_argument('--plot',        default='plots/resolution_sweep.png')
    p.add_argument('--device',      default='cuda')
    return p.parse_args()


def run_one_resolution(model, video_path: str, input_size: int,
                       device: str, conf: float, iou: float) -> dict:
    """Run full video at given resolution and return aggregated stats."""
    input_shape = (input_size, input_size)
    reader      = VideoReader(video_path)

    pre_ms, inf_ms, post_ms, total_ms_list = [], [], [], []
    det_counts = []

    with reader:
        while True:
            ret, frame = reader.read()
            if not ret:
                break

            with CUDATimer() as t_pre:
                img_lb, _, pad = letterbox(frame, input_shape)
                tensor = to_tensor(img_lb, device)

            with CUDATimer() as t_inf:
                with torch.no_grad():
                    raw = model(tensor)
                    if isinstance(raw, (list, tuple)):
                        raw = raw[0]

            with CUDATimer() as t_post:
                dets = run_nms(raw, conf, iou)
                det  = dets[0]
                if len(det):
                    det[:, :4] = scale_boxes(
                        det[:, :4],
                        orig_shape=(reader.height, reader.width),
                        input_shape=input_shape,
                        pad=pad,
                    )

            total = t_pre.elapsed_ms + t_inf.elapsed_ms + t_post.elapsed_ms
            pre_ms.append(t_pre.elapsed_ms)
            inf_ms.append(t_inf.elapsed_ms)
            post_ms.append(t_post.elapsed_ms)
            total_ms_list.append(total)
            det_counts.append(len(det))

    ts = compute_stats(total_ms_list)
    return {
        'resolution':      input_size,
        'frames':          len(total_ms_list),
        'mean_pre_ms':     round(compute_stats(pre_ms)['mean_ms'],  3),
        'mean_inf_ms':     round(compute_stats(inf_ms)['mean_ms'],  3),
        'mean_post_ms':    round(compute_stats(post_ms)['mean_ms'], 3),
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
    import numpy as np

    plt.rcParams.update({
        'figure.dpi': 150, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.alpha': 0.3,
    })

    labels    = [f'{r["resolution"]}×{r["resolution"]}' for r in rows]
    fps_vals  = [r['fps']             for r in rows]
    pre_vals  = [r['mean_pre_ms']     for r in rows]
    inf_vals  = [r['mean_inf_ms']     for r in rows]
    post_vals = [r['mean_post_ms']    for r in rows]
    det_vals  = [r['mean_detections'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # FPS per resolution
    bars = axes[0].bar(labels, fps_vals, color='#4c72b0', alpha=0.85, width=0.5)
    for bar, v in zip(bars, fps_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{v:.1f}', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_title('End-to-End FPS by Resolution')
    axes[0].set_ylabel('FPS (higher = faster)')
    axes[0].set_ylim(0, max(fps_vals) * 1.25)

    # Stacked latency breakdown
    axes[1].bar(labels, pre_vals,  label='preprocess', color='#66c2a5', alpha=0.85)
    axes[1].bar(labels, inf_vals,  bottom=pre_vals, label='inference', color='#fc8d62', alpha=0.85)
    bottom2 = [a + b for a, b in zip(pre_vals, inf_vals)]
    axes[1].bar(labels, post_vals, bottom=bottom2, label='postprocess', color='#e78ac3', alpha=0.85)
    axes[1].set_title('Stage Latency Breakdown')
    axes[1].set_ylabel('Mean latency (ms)')
    axes[1].legend(fontsize=8)

    # Mean detections per frame
    axes[2].plot(labels, det_vals, 'o-', color='#c44e52', linewidth=2, markersize=9)
    for x, y in zip(labels, det_vals):
        axes[2].annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=9)
    axes[2].set_title('Mean Detections per Frame\n(proxy for recall at conf=0.25)')
    axes[2].set_ylabel('Detections')
    axes[2].set_ylim(0, max(det_vals) * 1.4 if det_vals else 1)

    plt.suptitle('Resolution Sweep — PyTorch FP32', fontsize=12, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f'Saved → {plot_path}')


def main():
    args   = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'

    from ultralytics import YOLO
    yolo  = YOLO(args.model)
    model = yolo.model.to(device).eval()

    # Warmup at max resolution so GPU is fully warm before first timed run
    max_res = max(args.resolutions)
    print(f'Warming up at {max_res}×{max_res} ({args.warmup} iters) ...')
    dummy = torch.zeros(1, 3, max_res, max_res, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()

    rows = []
    for res in sorted(args.resolutions):
        print(f'\n── {res}×{res} ────────────────────────────────────────')
        row = run_one_resolution(model, args.video, res, device, args.conf, args.iou)
        rows.append(row)
        print(f'  FPS={row["fps"]:.1f}  inf={row["mean_inf_ms"]:.1f}ms  '
              f'pre={row["mean_pre_ms"]:.1f}ms  dets={row["mean_detections"]:.1f}')

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    with open(args.results, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Print table
    print(f'\n{"Res":<8} {"FPS":>6} {"inf_ms":>8} {"pre_ms":>8} {"post_ms":>9} {"dets":>6}')
    print('─' * 46)
    for r in rows:
        print(f'{r["resolution"]:<8} {r["fps"]:>6.1f} {r["mean_inf_ms"]:>8.2f} '
              f'{r["mean_pre_ms"]:>8.2f} {r["mean_post_ms"]:>9.2f} '
              f'{r["mean_detections"]:>6.1f}')
    print(f'\nSaved → {args.results}')

    plot_sweep(rows, args.plot)


if __name__ == '__main__':
    main()

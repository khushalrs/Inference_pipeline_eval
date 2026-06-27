"""Phase 1 + 2: PyTorch baseline inference pipeline.

Records per-frame latency for each pipeline stage:
  read → preprocess → inference → postprocess (NMS + box scale)

Usage:
    python3 scripts/run_pytorch_video.py \
        --video data/clip.mp4 \
        --results results/pytorch_raw_timings.csv
"""

import argparse
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.timing import CUDATimer
from src.video_io import load_frames
from src.preprocessing import to_tensor
from src.postprocessing import run_nms, scale_boxes
from src.metrics import compute_stats, fps_from_mean_ms


def parse_args():
    p = argparse.ArgumentParser(description='Phase 1: PyTorch baseline inference')
    p.add_argument('--video',        default='data/clip.mp4',
                   help='Input video path')
    p.add_argument('--results',      default='results/pytorch_raw_timings.csv',
                   help='Per-frame timing CSV')
    p.add_argument('--model',        default='yolo11n.pt',
                   help='Ultralytics model weights (downloaded automatically on first run)')
    p.add_argument('--input-size',   type=int,   default=640,
                   help='Square model input size')
    p.add_argument('--conf',         type=float, default=0.25,
                   help='Detection confidence threshold')
    p.add_argument('--iou',          type=float, default=0.45,
                   help='NMS IoU threshold')
    p.add_argument('--warmup',       type=int,   default=10,
                   help='Warmup forward passes before timing begins')
    p.add_argument('--device',       default='cuda',
                   help='Compute device: cuda or cpu')
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print('[WARNING] CUDA not available, falling back to CPU')

    input_shape = (args.input_size, args.input_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    print(f'Device      : {device}')
    print(f'Loading     : {args.model}')
    from ultralytics import YOLO
    yolo  = YOLO(args.model)
    model = yolo.model.to(device).eval()

    if device == 'cuda':
        print(f'GPU         : {torch.cuda.get_device_name(0)}')
        print(f'CUDA version: {torch.version.cuda}')

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f'Warming up ({args.warmup} iterations) ...')
    dummy = torch.zeros(1, 3, *input_shape, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()

    # ── Pre-buffer frames with letterbox (keeps RAM under ~3 GB for 4K source)
    print(f'\nVideo       : {args.video}')
    frame_buffer, src_fps, src_w, src_h = load_frames(args.video, input_shape=input_shape)
    n_frames = len(frame_buffer)

    FIELDS = ['frame', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms']
    rows = []

    for frame_idx, (img_lb, pad) in enumerate(frame_buffer):

        # Stage 1: BGR→RGB + normalize + move to GPU (letterbox already done during buffering)
        with CUDATimer() as t_pre:
            tensor = to_tensor(img_lb, device)

        # Stage 2: model forward pass
        with CUDATimer() as t_inf:
            with torch.no_grad():
                raw_preds = model(tensor)
                if isinstance(raw_preds, (list, tuple)):
                    raw_preds = raw_preds[0]

        # Stage 3: NMS + scale boxes to original frame coords
        with CUDATimer() as t_post:
            dets = run_nms(raw_preds, args.conf, args.iou)
            det  = dets[0]
            if len(det):
                det[:, :4] = scale_boxes(
                    det[:, :4],
                    orig_shape=(src_h, src_w),
                    input_shape=input_shape,
                    pad=pad,
                )

        total_ms = t_pre.elapsed_ms + t_inf.elapsed_ms + t_post.elapsed_ms

        rows.append({
            'frame':          frame_idx,
            'preprocess_ms':  round(t_pre.elapsed_ms,   3),
            'inference_ms':   round(t_inf.elapsed_ms,   3),
            'postprocess_ms': round(t_post.elapsed_ms,  3),
            'total_ms':       round(total_ms,            3),
        })

        if frame_idx % 50 == 0 or frame_idx == 1:
            print(f'  Frame {frame_idx:>4}/{n_frames}  '
                  f'total={total_ms:6.1f}ms  '
                  f'inf={t_inf.elapsed_ms:5.1f}ms  '
                  f'pre={t_pre.elapsed_ms:5.1f}ms  '
                  f'post={t_post.elapsed_ms:5.1f}ms')

    # ── Save CSV ─────────────────────────────────────────────────────────────
    with open(args.results, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # ── Summary (exclude warmup frames from printed stats) ───────────────────
    total_times = [r['total_ms'] for r in rows[args.warmup:]]
    stats       = compute_stats(total_times)

    print(f'\n{"─"*50}')
    print(f'Frames processed : {n_frames}  (stats exclude first {args.warmup} warmup frames)')
    print(f'Mean total       : {stats["mean_ms"]:.2f} ms')
    print(f'p50 / p90 / p99  : {stats["p50_ms"]:.2f} / {stats["p90_ms"]:.2f} / {stats["p99_ms"]:.2f} ms')
    print(f'Inference FPS    : {fps_from_mean_ms(stats["mean_ms"]):.1f}')
    print(f'{"─"*50}')
    print(f'Timings saved    : {args.results}')


if __name__ == '__main__':
    main()

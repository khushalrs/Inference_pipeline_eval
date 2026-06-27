"""Batch 02: torch.compile (reduce-overhead) inference pipeline on driving video.

Compiles the YOLO11n backbone+neck with mode='reduce-overhead' (CUDA Graphs)
for maximum kernel launch savings. The Detect head runs in eager mode because
it mutates self.anchors/self.strides on every forward pass, which is
incompatible with CUDA Graph static memory requirements.

Split: model.model[:-1] → compiled backbone (CUDA Graphs)
       model.model[-1]  → Detect head (eager, ~0.1ms, not the bottleneck)

torch.compile requires PyTorch >= 2.0 and a CUDA GPU.

Usage:
    python3 scripts/run_compile_video.py \
        --video   data/clip.mp4 \
        --results results/b02_compile_raw_timings.csv \
        --model   yolo11n.pt \
        --device  cuda
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
    p = argparse.ArgumentParser(description='Batch 02: torch.compile inference')
    p.add_argument('--video',       default='data/clip.mp4')
    p.add_argument('--results',     default='results/b02_compile_raw_timings.csv')
    p.add_argument('--model',       default='yolo11n.pt')
    p.add_argument('--input-size',  type=int,   default=640)
    p.add_argument('--conf',        type=float, default=0.25)
    p.add_argument('--iou',         type=float, default=0.45)
    p.add_argument('--warmup',      type=int,   default=20,
                   help='Warmup forward passes — must be enough for compile to finish (default: 20)')
    p.add_argument('--device',      default='cuda')
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print('[WARNING] CUDA not available, falling back to CPU')

    if not hasattr(torch, 'compile'):
        print('[ERROR] torch.compile requires PyTorch >= 2.0')
        sys.exit(1)

    input_shape = (args.input_size, args.input_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    print(f'Device      : {device}')
    print(f'Loading     : {args.model}')
    from ultralytics import YOLO
    yolo       = YOLO(args.model)
    full_model = yolo.model.to(device).eval()

    # Compile backbone+neck only — the Detect head (last module) mutates
    # self.anchors/self.strides each forward pass, incompatible with CUDA Graphs.
    backbone = torch.compile(full_model.model[:-1], mode='reduce-overhead')
    head     = full_model.model[-1]

    if device == 'cuda':
        print(f'GPU         : {torch.cuda.get_device_name(0)}')
        print(f'PyTorch     : {torch.__version__}')
    print('Backbone compiled with reduce-overhead (CUDA Graphs). Head runs in eager.')

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f'Warming up ({args.warmup} iterations — CUDA Graph capture on first pass) ...')
    dummy = torch.zeros(1, 3, *input_shape, device=device)
    with torch.no_grad():
        for _ in range(args.warmup):
            features  = backbone(dummy)
            _         = head(features)
    if device == 'cuda':
        torch.cuda.synchronize()
    print('Warmup complete.\n')

    # ── Pre-buffer frames with letterbox (keeps RAM under ~3 GB for 4K source)
    print(f'Video       : {args.video}')
    frame_buffer, src_fps, src_w, src_h = load_frames(args.video, input_shape=input_shape)
    n_frames = len(frame_buffer)

    FIELDS = ['frame', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms']
    rows = []

    for frame_idx, (img_lb, pad) in enumerate(frame_buffer):

        # BGR→RGB + normalize + move to GPU (letterbox done during buffering)
        with CUDATimer() as t_pre:
            tensor = to_tensor(img_lb, device)

        with CUDATimer() as t_inf:
            with torch.no_grad():
                features  = backbone(tensor)
                raw_preds = head(features)
                if isinstance(raw_preds, (list, tuple)):
                    raw_preds = raw_preds[0]

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

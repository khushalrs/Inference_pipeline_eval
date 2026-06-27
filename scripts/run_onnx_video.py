"""Phase 3: ONNX Runtime inference pipeline on driving video.

Mirrors run_pytorch_video.py exactly — same 4 timed stages, same output CSV
schema — so Phase 2 benchmark.py and plot_results.py work unchanged.

The only difference from the PyTorch script is the inference stage:
  - input is a numpy float32 array instead of a torch tensor
  - inference uses onnxruntime.InferenceSession instead of model.forward()
  - raw output is converted back to a torch tensor before NMS so all
    postprocessing code is reused without modification

Usage:
    python3 scripts/run_onnx_video.py \
        --video      data/clip.mp4 \
        --onnx-model models/yolo11n.onnx \
        --results    results/onnx_raw_timings.csv \
        --device     cuda
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.timing import CUDATimer
from src.video_io import load_frames
from src.preprocessing import to_numpy_input
from src.postprocessing import run_nms, scale_boxes
from src.metrics import compute_stats, fps_from_mean_ms


def parse_args():
    p = argparse.ArgumentParser(description='Phase 3: ONNX Runtime inference')
    p.add_argument('--video',        default='data/clip.mp4')
    p.add_argument('--onnx-model',   default='models/yolo11n.onnx',
                   help='Path to exported ONNX model')
    p.add_argument('--results',      default='results/onnx_raw_timings.csv')
    p.add_argument('--input-size',   type=int,   default=640)
    p.add_argument('--conf',         type=float, default=0.25)
    p.add_argument('--iou',          type=float, default=0.45)
    p.add_argument('--warmup',       type=int,   default=10)
    p.add_argument('--device',       default='cuda',
                   help='cuda or cpu — selects ORT execution provider')
    return p.parse_args()


def build_ort_session(onnx_path: str, device: str):
    import onnxruntime as ort
    available = ort.get_available_providers()
    if device == 'cuda' and 'CUDAExecutionProvider' in available:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            print('[WARNING] CUDAExecutionProvider not available — running on CPU. '
                  'Install onnxruntime-gpu to enable GPU inference.')
    session = ort.InferenceSession(onnx_path, providers=providers)
    active_ep = session.get_providers()[0]
    print(f'ORT execution provider : {active_ep}')
    if active_ep == 'CPUExecutionProvider' and device == 'cuda':
        print('[WARNING] Running on CPU EP despite --device cuda. '
              'Results will not be comparable to TensorRT GPU numbers.')
    return session


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print('[WARNING] CUDA not available, falling back to CPU')

    input_shape = (args.input_size, args.input_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)

    # ── ORT session ──────────────────────────────────────────────────────────
    if not os.path.exists(args.onnx_model):
        print(f'[ERROR] ONNX model not found: {args.onnx_model}')
        print('        Run scripts/export_onnx.py first.')
        sys.exit(1)

    print(f'Loading ONNX model: {args.onnx_model}')
    session    = build_ort_session(args.onnx_model, device)
    input_name = session.get_inputs()[0].name

    if device == 'cuda':
        print(f'GPU : {torch.cuda.get_device_name(0)}')

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f'Warming up ({args.warmup} iterations) ...')
    dummy_np = np.zeros((1, 3, *input_shape), dtype=np.float32)
    for _ in range(args.warmup):
        session.run(None, {input_name: dummy_np})
    if device == 'cuda':
        torch.cuda.synchronize()

    # ── Pre-buffer frames with letterbox (keeps RAM under ~3 GB for 4K source)
    print(f'\nVideo      : {args.video}')
    frame_buffer, src_fps, src_w, src_h = load_frames(args.video, input_shape=input_shape)
    n_frames = len(frame_buffer)

    FIELDS = ['frame', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms']
    rows = []

    for frame_idx, (img_lb, pad) in enumerate(frame_buffer):

        # Stage 1: BGR→RGB + normalize → numpy float32 (letterbox done during buffering)
        with CUDATimer() as t_pre:
            arr_np = to_numpy_input(img_lb)

        # Stage 2: ORT inference
        with CUDATimer() as t_inf:
            raw_np = session.run(None, {input_name: arr_np})[0]
            raw_t  = torch.from_numpy(raw_np)

        # Stage 3: NMS + scale boxes
        with CUDATimer() as t_post:
            dets = run_nms(raw_t, args.conf, args.iou)
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

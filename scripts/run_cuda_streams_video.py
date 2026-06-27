"""Batch 03: CUDA Streams overlapped inference pipeline.

Overlaps three stages across consecutive frames using two CUDA streams:
  - copy_stream : async host→device transfer for frame N+1
  - infer_stream: TRT inference on frame N

Timeline (serial vs overlapped):
  Serial:    [pre N][copy N][inf N][post N] | [pre N+1][copy N+1][inf N+1][post N+1]
  Overlapped:[pre N][copy N][inf N][post N]
                             [copy N+1]      ← overlaps with inf N on GPU
                                      [inf N+1][post N+1]

Pinned (page-locked) host memory is required for async H→D to actually overlap
with GPU compute — pageable memory forces synchronous copies.

Expected gain on T4: 25–40% throughput improvement over serialised TRT FP16.

Works only with TRT engines (FP16 recommended — fastest inference = most
headroom for overlap to matter).

Usage:
    python3 scripts/run_cuda_streams_video.py \
        --video   data/clip.mp4 \
        --engine  models/yolo11n_fp16.engine \
        --results results/b03_cuda_streams_raw_timings.csv
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_io import load_frames
from src.preprocessing import to_numpy_input
from src.postprocessing import run_nms, scale_boxes
from src.metrics import compute_stats, fps_from_mean_ms
from src.trt_runner import TRTRunner


def parse_args():
    p = argparse.ArgumentParser(description='Batch 03: CUDA Streams overlapped pipeline')
    p.add_argument('--video',      default='data/clip.mp4')
    p.add_argument('--engine',     default='models/yolo11n_fp16.engine',
                   help='TRT engine (.engine) — FP16 recommended for maximum overlap benefit')
    p.add_argument('--results',    default='results/b03_cuda_streams_raw_timings.csv')
    p.add_argument('--input-size', type=int,   default=640)
    p.add_argument('--conf',       type=float, default=0.25)
    p.add_argument('--iou',        type=float, default=0.45)
    p.add_argument('--warmup',     type=int,   default=20)
    return p.parse_args()


# ── Pinned host buffer helpers ────────────────────────────────────────────────

def make_pinned(shape) -> torch.Tensor:
    """Allocate a pinned (page-locked) host tensor for async H→D transfers."""
    return torch.zeros(shape, dtype=torch.float32).pin_memory()


def preprocess_to_pinned(img_bgr: np.ndarray, buf: torch.Tensor):
    """BGR uint8 → normalised float32, written into a pinned host tensor in-place."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    arr     = np.ascontiguousarray(img_rgb.transpose(2, 0, 1)).astype(np.float32) / 255.0
    buf.copy_(torch.from_numpy(arr))    # CPU-side copy into pinned memory


# ── CUDA-stream-aware TRT inference ──────────────────────────────────────────

def infer_on_stream(runner: TRTRunner, gpu_buf: torch.Tensor,
                    stream: torch.cuda.Stream) -> torch.Tensor:
    """Run TRT inference on an already-resident GPU tensor using the given stream."""
    runner._in_buf.copy_(gpu_buf, non_blocking=True)
    with torch.cuda.stream(stream):
        cuda_stream = stream.cuda_stream
        from tensorrt import __version__ as _trtv
        if int(_trtv.split('.')[0]) >= 10:
            runner.context.execute_async_v3(cuda_stream)
        else:
            runner.context.execute_async_v2(runner._bindings, cuda_stream)
    out = runner._out_buf
    return out if out.dtype == torch.float32 else out.float()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print('[ERROR] CUDA GPU required.')
        sys.exit(1)

    if not os.path.exists(args.engine):
        print(f'[ERROR] Engine not found: {args.engine}')
        sys.exit(1)

    input_shape = (args.input_size, args.input_size)
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)

    print(f'Engine     : {args.engine}')
    print(f'GPU        : {torch.cuda.get_device_name(0)}')
    runner = TRTRunner(args.engine, device='cuda')
    print(f'Input      : {runner.input_shape}')
    print(f'Output     : {runner.output_shape}')

    # ── Two CUDA streams ─────────────────────────────────────────────────────
    copy_stream  = torch.cuda.Stream()   # async H→D transfer
    infer_stream = torch.cuda.Stream()   # TRT execution
    copy_done    = torch.cuda.Event()    # signals H→D complete → infer_stream can start
    infer_done   = torch.cuda.Event()    # signals inference complete → CPU can read output

    # ── Ping-pong GPU buffers (frame N and N+1 in flight simultaneously) ─────
    C, H, W = 3, args.input_size, args.input_size
    host_bufs = [make_pinned((C, H, W)), make_pinned((C, H, W))]   # pinned host
    gpu_bufs  = [
        torch.zeros(1, C, H, W, dtype=torch.float32, device='cuda').contiguous(),
        torch.zeros(1, C, H, W, dtype=torch.float32, device='cuda').contiguous(),
    ]

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f'\nWarming up ({args.warmup} iterations) ...')
    dummy_np = np.zeros((1, C, H, W), dtype=np.float32)
    for _ in range(args.warmup):
        runner.infer(dummy_np)
    torch.cuda.synchronize()
    print('Warmup complete.\n')

    # ── Pre-buffer frames ────────────────────────────────────────────────────
    print(f'Video      : {args.video}')
    frame_buffer, src_fps, src_w, src_h = load_frames(args.video, input_shape=input_shape)
    n_frames = len(frame_buffer)

    FIELDS = ['frame', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_ms']
    rows   = []

    # ── Wall-clock throughput tracking ───────────────────────────────────────
    wall_start = time.perf_counter()

    for frame_idx, (img_lb, pad) in enumerate(frame_buffer):
        curr = frame_idx % 2
        t0   = time.perf_counter()

        # Stage 1: CPU preprocessing → pinned host buffer (curr slot)
        pre_start = time.perf_counter()
        preprocess_to_pinned(img_lb, host_bufs[curr])
        pre_ms = (time.perf_counter() - pre_start) * 1000

        # Stage 2a: Async H→D copy of current frame on copy_stream
        with torch.cuda.stream(copy_stream):
            gpu_bufs[curr].copy_(host_bufs[curr].unsqueeze(0), non_blocking=True)
            copy_done.record(copy_stream)

        # Stage 2b: TRT inference on infer_stream (waits for copy_done first)
        inf_start_event  = torch.cuda.Event(enable_timing=True)
        inf_end_event    = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(infer_stream):
            infer_stream.wait_event(copy_done)
            inf_start_event.record(infer_stream)
            runner._in_buf.copy_(gpu_bufs[curr], non_blocking=True)
            cuda_stream = infer_stream.cuda_stream
            import tensorrt as _trt
            if int(_trt.__version__.split('.')[0]) >= 10:
                runner.context.execute_async_v3(cuda_stream)
            else:
                runner.context.execute_async_v2(runner._bindings, cuda_stream)
            inf_end_event.record(infer_stream)
            infer_done.record(infer_stream)

        # Stage 3: Wait for inference, then NMS (CPU)
        infer_done.synchronize()
        inf_ms = inf_start_event.elapsed_time(inf_end_event)

        raw_preds = runner._out_buf if runner._out_buf.dtype == torch.float32 \
                    else runner._out_buf.float()
        if isinstance(raw_preds, (list, tuple)):
            raw_preds = raw_preds[0]

        post_start = time.perf_counter()
        dets = run_nms(raw_preds, args.conf, args.iou)
        det  = dets[0]
        if len(det):
            det[:, :4] = scale_boxes(
                det[:, :4],
                orig_shape  = (src_h, src_w),
                input_shape = input_shape,
                pad         = pad,
            )
        post_ms = (time.perf_counter() - post_start) * 1000

        total_ms = (time.perf_counter() - t0) * 1000

        rows.append({
            'frame':          frame_idx,
            'preprocess_ms':  round(pre_ms,    3),
            'inference_ms':   round(inf_ms,     3),
            'postprocess_ms': round(post_ms,    3),
            'total_ms':       round(total_ms,   3),
        })

        if frame_idx % 50 == 0 or frame_idx == 1:
            print(f'  Frame {frame_idx:>4}/{n_frames}  '
                  f'total={total_ms:6.1f}ms  '
                  f'inf={inf_ms:5.1f}ms  '
                  f'pre={pre_ms:5.1f}ms  '
                  f'post={post_ms:5.1f}ms')

    wall_elapsed = time.perf_counter() - wall_start
    wall_fps     = n_frames / wall_elapsed

    # ── Save CSV ─────────────────────────────────────────────────────────────
    with open(args.results, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    measured_rows = rows[args.warmup:]
    total_times   = [r['total_ms'] for r in measured_rows]
    stats         = compute_stats(total_times)

    print(f'\n{"─"*55}')
    print(f'Frames processed  : {n_frames}  (stats exclude first {args.warmup} warmup frames)')
    print(f'Mean total        : {stats["mean_ms"]:.2f} ms')
    print(f'p50 / p90 / p99   : {stats["p50_ms"]:.2f} / {stats["p90_ms"]:.2f} / {stats["p99_ms"]:.2f} ms')
    print(f'Per-frame FPS     : {fps_from_mean_ms(stats["mean_ms"]):.1f}')
    print(f'Wall-clock FPS    : {wall_fps:.1f}  ← includes stream overlap benefit')
    print(f'{"─"*55}')
    print(f'Timings saved     : {args.results}')
    print()
    print('Compare wall-clock FPS vs run_tensorrt_video.py to quantify overlap gain.')


if __name__ == '__main__':
    main()

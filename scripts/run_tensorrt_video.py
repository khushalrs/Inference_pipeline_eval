"""Phase 4: TensorRT inference pipeline on driving video.

Identical 5-stage timing structure to run_pytorch_video.py and run_onnx_video.py.
Use this script for both FP32 and FP16 engines by changing --engine.

The TRTRunner uses PyTorch CUDA tensors as GPU memory — no pycuda needed.
CUDATimer correctly captures inference time because TRT runs on the same
PyTorch CUDA stream and is flushed by torch.cuda.synchronize().

Usage:
    # FP32
    python3 scripts/run_tensorrt_video.py \
        --video        data/clip.mp4 \
        --engine       models/yolo11n_fp32.engine \
        --output-video results/trt_fp32_output.mp4 \
        --results      results/trt_fp32_raw_timings.csv

    # FP16
    python3 scripts/run_tensorrt_video.py \
        --video        data/clip.mp4 \
        --engine       models/yolo11n_fp16.engine \
        --output-video results/trt_fp16_output.mp4 \
        --results      results/trt_fp16_raw_timings.csv
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.timing import CUDATimer
from src.video_io import VideoReader, VideoWriter
from src.preprocessing import letterbox, to_numpy_input
from src.postprocessing import run_nms, scale_boxes, draw_detections
from src.metrics import compute_stats, fps_from_mean_ms
from src.trt_runner import TRTRunner


def parse_args():
    p = argparse.ArgumentParser(description='Phase 4: TensorRT video inference')
    p.add_argument('--video',        default='data/clip.mp4')
    p.add_argument('--engine',       required=True,
                   help='Path to .engine file (from build_tensorrt_engine.py)')
    p.add_argument('--pt-model',     default='yolo11n.pt',
                   help='YOLO weights — only used to load class names')
    p.add_argument('--output-video', default='results/trt_output.mp4')
    p.add_argument('--results',      default='results/trt_raw_timings.csv')
    p.add_argument('--input-size',   type=int,   default=640)
    p.add_argument('--conf',         type=float, default=0.25)
    p.add_argument('--iou',          type=float, default=0.45)
    p.add_argument('--warmup',       type=int,   default=10,
                   help='Warmup forward passes before timing begins')
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print('[ERROR] TensorRT requires a CUDA GPU. Colab: Runtime → Change runtime type → GPU.')
        sys.exit(1)

    if not os.path.exists(args.engine):
        print(f'[ERROR] Engine not found: {args.engine}')
        print('        Run scripts/build_tensorrt_engine.py first.')
        sys.exit(1)

    input_shape = (args.input_size, args.input_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.results)),      exist_ok=True)

    # ── TRT runner ───────────────────────────────────────────────────────────
    print(f'Loading engine : {args.engine}')
    runner = TRTRunner(args.engine, device='cuda')
    print(f'GPU            : {torch.cuda.get_device_name(0)}')
    print(f'Input  shape   : {runner.input_shape}')
    print(f'Output shape   : {runner.output_shape}')
    print(f'Output dtype   : {runner._out_buf.dtype}')  # float32 or float16

    from ultralytics import YOLO
    class_names = YOLO(args.pt_model).names

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f'\nWarming up ({args.warmup} iterations) ...')
    dummy_np = np.zeros((1, 3, *input_shape), dtype=np.float32)
    for _ in range(args.warmup):
        runner.infer(dummy_np)
    torch.cuda.synchronize()
    print('Warmup complete.\n')

    # ── Inference loop ───────────────────────────────────────────────────────
    reader = VideoReader(args.video)
    print(f'Video   : {args.video}')
    print(f'Frames  : {reader.frame_count}  |  {reader.width}x{reader.height}  |  {reader.fps:.1f} FPS')
    print(f'Output  : {args.output_video}\n')

    FIELDS = ['frame', 'read_ms', 'preprocess_ms', 'inference_ms',
              'postprocess_ms', 'draw_write_ms', 'total_ms']
    rows = []

    with reader, VideoWriter(args.output_video, reader.fps, reader.width, reader.height) as writer:
        frame_idx = 0
        while True:

            # Stage 1: read frame
            with CUDATimer() as t_read:
                ret, frame = reader.read()
            if not ret:
                break

            # Stage 2: letterbox + convert to float32 numpy (CPU ops)
            with CUDATimer() as t_pre:
                img_lb, ratio, pad = letterbox(frame, input_shape)
                arr_np = to_numpy_input(img_lb)

            # Stage 3: host→device copy + TRT execution (synced by CUDATimer exit)
            with CUDATimer() as t_inf:
                raw_t = runner.infer(arr_np)

            # Stage 4: NMS + scale boxes back to original frame coords
            with CUDATimer() as t_post:
                dets = run_nms(raw_t, args.conf, args.iou)
                det  = dets[0]
                if len(det):
                    det[:, :4] = scale_boxes(
                        det[:, :4],
                        orig_shape=(reader.height, reader.width),
                        input_shape=input_shape,
                        pad=pad,
                    )

            # Stage 5: draw detections + write to output video
            with CUDATimer() as t_draw:
                annotated = draw_detections(frame.copy(), det, class_names)
                writer.write(annotated)

            total_ms = (t_read.elapsed_ms + t_pre.elapsed_ms +
                        t_inf.elapsed_ms   + t_post.elapsed_ms + t_draw.elapsed_ms)

            rows.append({
                'frame':          frame_idx,
                'read_ms':        round(t_read.elapsed_ms,  3),
                'preprocess_ms':  round(t_pre.elapsed_ms,   3),
                'inference_ms':   round(t_inf.elapsed_ms,   3),
                'postprocess_ms': round(t_post.elapsed_ms,  3),
                'draw_write_ms':  round(t_draw.elapsed_ms,  3),
                'total_ms':       round(total_ms,            3),
            })

            frame_idx += 1
            if frame_idx % 50 == 0 or frame_idx == 1:
                print(f'  Frame {frame_idx:>4}/{reader.frame_count}  '
                      f'total={total_ms:6.1f}ms  '
                      f'inf={t_inf.elapsed_ms:5.1f}ms  '
                      f'pre={t_pre.elapsed_ms:5.1f}ms  '
                      f'post={t_post.elapsed_ms:5.1f}ms')

    # ── Save CSV ─────────────────────────────────────────────────────────────
    with open(args.results, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    total_times = [r['total_ms'] for r in rows]
    stats       = compute_stats(total_times)

    print(f'\n{"─"*50}')
    print(f'Frames processed : {frame_idx}')
    print(f'Mean total       : {stats["mean_ms"]:.2f} ms')
    print(f'p50 / p90 / p99  : {stats["p50_ms"]:.2f} / {stats["p90_ms"]:.2f} / {stats["p99_ms"]:.2f} ms')
    print(f'End-to-end FPS   : {fps_from_mean_ms(stats["mean_ms"]):.1f}')
    print(f'{"─"*50}')
    print(f'Timings saved    : {args.results}')
    print(f'Output video     : {args.output_video}')


if __name__ == '__main__':
    main()

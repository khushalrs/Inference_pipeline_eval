"""Phase 3: ONNX Runtime inference pipeline on driving video.

Mirrors run_pytorch_video.py exactly — same 5 timed stages, same output CSV
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
        --pt-model   yolo11n.pt \
        --output-video results/onnx_output.mp4 \
        --results      results/onnx_raw_timings.csv \
        --device       cuda
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


def parse_args():
    p = argparse.ArgumentParser(description='Phase 3: ONNX Runtime inference')
    p.add_argument('--video',        default='data/clip.mp4')
    p.add_argument('--onnx-model',   default='models/yolo11n.onnx',
                   help='Path to exported ONNX model')
    p.add_argument('--pt-model',     default='yolo11n.pt',
                   help='Original PyTorch weights — used only to retrieve class names')
    p.add_argument('--output-video', default='results/onnx_output.mp4')
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
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers()
                 else ['CPUExecutionProvider'])
    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f'ORT providers : {session.get_providers()}')
    return session


def main():
    args = parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if device == 'cpu' and args.device == 'cuda':
        print('[WARNING] CUDA not available, falling back to CPU')

    input_shape = (args.input_size, args.input_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.results)),      exist_ok=True)

    # ── ORT session ──────────────────────────────────────────────────────────
    if not os.path.exists(args.onnx_model):
        print(f'[ERROR] ONNX model not found: {args.onnx_model}')
        print('        Run scripts/export_onnx.py first.')
        sys.exit(1)

    print(f'Loading ONNX model: {args.onnx_model}')
    session    = build_ort_session(args.onnx_model, device)
    input_name = session.get_inputs()[0].name

    # Class names come from the original PyTorch weights
    from ultralytics import YOLO
    class_names = YOLO(args.pt_model).names

    if device == 'cuda':
        print(f'GPU : {torch.cuda.get_device_name(0)}')

    # ── Warmup ───────────────────────────────────────────────────────────────
    print(f'Warming up ({args.warmup} iterations) ...')
    dummy_np = np.zeros((1, 3, *input_shape), dtype=np.float32)
    for _ in range(args.warmup):
        session.run(None, {input_name: dummy_np})
    if device == 'cuda':
        torch.cuda.synchronize()

    # ── Inference loop ───────────────────────────────────────────────────────
    reader = VideoReader(args.video)
    print(f'\nVideo      : {args.video}')
    print(f'Resolution : {reader.width}x{reader.height}  |  FPS: {reader.fps:.1f}  |  Frames: {reader.frame_count}')
    print(f'Output     : {args.output_video}\n')

    FIELDS = ['frame', 'read_ms', 'preprocess_ms', 'inference_ms',
              'postprocess_ms', 'draw_write_ms', 'total_ms']
    rows = []

    with reader, VideoWriter(args.output_video, reader.fps, reader.width, reader.height) as writer:
        frame_idx = 0
        while True:

            # Stage 1: read
            with CUDATimer() as t_read:
                ret, frame = reader.read()
            if not ret:
                break

            # Stage 2: preprocess — letterbox + BGR→RGB + normalize → numpy float32
            with CUDATimer() as t_pre:
                img_lb, ratio, pad = letterbox(frame, input_shape)
                arr_np = to_numpy_input(img_lb)

            # Stage 3: ORT inference
            with CUDATimer() as t_inf:
                raw_np = session.run(None, {input_name: arr_np})[0]
                # Convert to torch tensor so NMS code is identical to PyTorch pipeline
                raw_t  = torch.from_numpy(raw_np)

            # Stage 4: NMS + scale boxes
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

            # Stage 5: draw + write
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

    # ── Summary ──────────────────────────────────────────────────────────────
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

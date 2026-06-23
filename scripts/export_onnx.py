"""Phase 3: Export YOLO11n to ONNX and validate output consistency.

Exports the PyTorch model to ONNX using the ultralytics built-in exporter,
then validates consistency against the original PyTorch model on N sampled
frames from the input video, reporting:
  - Raw prediction max/mean absolute difference
  - Post-NMS detection count match rate
  - Bounding box coordinate drift
  - Confidence score drift

Usage:
    python3 scripts/export_onnx.py \
        --model      yolo11n.pt \
        --output     models/yolo11n.onnx \
        --input-size 640 \
        --validate   \
        --video      data/clip.mp4 \
        --val-frames 20
"""

import argparse
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import letterbox, to_tensor, to_numpy_input
from src.postprocessing import run_nms, scale_boxes


def parse_args():
    p = argparse.ArgumentParser(description='Phase 3: ONNX export and validation')
    p.add_argument('--model',       default='yolo11n.pt',
                   help='Ultralytics YOLO weights to export')
    p.add_argument('--output',      default='models/yolo11n.onnx',
                   help='Destination path for the exported ONNX file')
    p.add_argument('--input-size',  type=int, default=640,
                   help='Square model input size')
    p.add_argument('--validate',    action='store_true',
                   help='Run PyTorch vs ONNX output comparison after export')
    p.add_argument('--video',       default='data/clip.mp4',
                   help='Video to sample validation frames from')
    p.add_argument('--val-frames',  type=int, default=20,
                   help='Number of frames to use for validation')
    p.add_argument('--conf',        type=float, default=0.25)
    p.add_argument('--iou',         type=float, default=0.45)
    p.add_argument('--device',      default='cuda')
    return p.parse_args()


# ── Export ────────────────────────────────────────────────────────────────────

def export_model(model_path: str, output_path: str, input_size: int) -> str:
    from ultralytics import YOLO

    print(f'Exporting {model_path} → ONNX (imgsz={input_size}) ...')
    yolo     = YOLO(model_path)
    exported = yolo.export(format='onnx', imgsz=input_size, dynamic=False, simplify=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if os.path.abspath(str(exported)) != os.path.abspath(output_path):
        shutil.move(str(exported), output_path)

    size_mb = os.path.getsize(output_path) / 1e6
    print(f'Saved  → {output_path}  ({size_mb:.1f} MB)')
    return output_path


# ── Validation helpers ────────────────────────────────────────────────────────

def _sample_frames(video_path: str, n: int) -> list:
    """Return n evenly-spaced BGR frames from the video."""
    import cv2
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total // n)
    frames = []
    for idx in range(0, min(total, n * stride), stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) == n:
            break
    cap.release()
    return frames


def _pt_predict(model, tensor, conf, iou, input_shape, orig_shape, pad):
    with torch.no_grad():
        raw = model(tensor)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
    dets = run_nms(raw, conf, iou)
    det  = dets[0]
    if len(det):
        det[:, :4] = scale_boxes(det[:, :4], orig_shape, input_shape, pad)
    return raw.cpu().numpy(), det


def _ort_predict(session, input_name, arr_np, conf, iou, input_shape, orig_shape, pad):
    raw_np = session.run(None, {input_name: arr_np})[0]
    raw_t  = torch.from_numpy(raw_np)
    dets   = run_nms(raw_t, conf, iou)
    det    = dets[0]
    if len(det):
        det[:, :4] = scale_boxes(det[:, :4], orig_shape, input_shape, pad)
    return raw_np, det


def validate(onnx_path: str, pt_model_path: str, video_path: str,
             n_frames: int, input_size: int, conf: float, iou: float, device: str):
    import onnxruntime as ort
    from ultralytics import YOLO

    device = device if torch.cuda.is_available() else 'cpu'
    input_shape = (input_size, input_size)

    print(f'\n── Validation ({n_frames} frames) ──────────────────────────────')
    print(f'PyTorch model : {pt_model_path}')
    print(f'ONNX model    : {onnx_path}')

    # Load PyTorch model
    yolo    = YOLO(pt_model_path)
    pt_net  = yolo.model.to(device).eval()
    class_names = yolo.names

    # Load ONNX Runtime session
    providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                 if device == 'cuda' else ['CPUExecutionProvider'])
    session    = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name
    print(f'ORT providers : {session.get_providers()}')

    frames = _sample_frames(video_path, n_frames)
    if not frames:
        print('[ERROR] No frames sampled — check video path.')
        return

    raw_diffs, conf_diffs, box_diffs = [], [], []
    count_matches = 0

    for i, frame in enumerate(frames):
        orig_shape = (frame.shape[0], frame.shape[1])
        img_lb, ratio, pad = letterbox(frame, input_shape)

        tensor  = to_tensor(img_lb, device)
        arr_np  = to_numpy_input(img_lb)

        pt_raw, pt_det   = _pt_predict(pt_net, tensor, conf, iou, input_shape, orig_shape, pad)
        ort_raw, ort_det = _ort_predict(session, input_name, arr_np, conf, iou, input_shape, orig_shape, pad)

        # Raw tensor diff
        raw_diffs.append(float(np.abs(pt_raw - ort_raw).max()))

        # Detection-level comparison
        pt_n  = len(pt_det)
        ort_n = len(ort_det)
        if pt_n == ort_n:
            count_matches += 1
            if pt_n > 0:
                # Sort both by confidence descending to align detections
                pt_sorted  = pt_det[pt_det[:, 4].argsort(descending=True)].cpu().numpy()
                ort_sorted = ort_det[ort_det[:, 4].argsort(descending=True)].cpu().numpy()
                conf_diffs.append(float(np.abs(pt_sorted[:, 4] - ort_sorted[:, 4]).mean()))
                box_diffs.append(float(np.abs(pt_sorted[:, :4] - ort_sorted[:, :4]).mean()))

    # ── Report ────────────────────────────────────────────────────────────────
    n = len(frames)
    print(f'\n{"Metric":<40} {"Value":>12}')
    print('─' * 54)
    print(f'{"Frames compared":<40} {n:>12}')
    print(f'{"Raw output max abs diff":<40} {max(raw_diffs):>12.6f}')
    print(f'{"Raw output mean max abs diff":<40} {np.mean(raw_diffs):>12.6f}')
    print(f'{"Detection count match rate":<40} {count_matches/n*100:>11.1f}%')
    if conf_diffs:
        print(f'{"Mean confidence score diff":<40} {np.mean(conf_diffs):>12.6f}')
        print(f'{"Mean bounding box coord diff (px)":<40} {np.mean(box_diffs):>12.3f}')
    else:
        print(f'{"Conf / box diff":<40} {"N/A — no matched detections":>12}')

    if max(raw_diffs) < 1e-3:
        print('\n[PASS] Raw outputs are numerically consistent (max diff < 1e-3).')
    else:
        print(f'\n[WARN] Max raw diff = {max(raw_diffs):.6f}. Check precision settings.')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    onnx_path = export_model(args.model, args.output, args.input_size)

    if args.validate:
        validate(
            onnx_path      = onnx_path,
            pt_model_path  = args.model,
            video_path     = args.video,
            n_frames       = args.val_frames,
            input_size     = args.input_size,
            conf           = args.conf,
            iou            = args.iou,
            device         = args.device,
        )


if __name__ == '__main__':
    main()

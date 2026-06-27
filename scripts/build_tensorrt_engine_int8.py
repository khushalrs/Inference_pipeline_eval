"""Batch 03: Build TensorRT INT8 engine with entropy calibration.

Requires calibration frames extracted by extract_calibration_frames.py.
Calibration reads activation statistics across the calibration set to find
optimal INT8 scale factors per layer (IInt8EntropyCalibrator2).

The calibration cache is saved to models/calibration.cache and can be reused
to rebuild the engine in a new Colab session without re-running calibration.
Copy both the .engine and .cache to Google Drive to persist across sessions.

Build times on T4:  INT8 ~ 10–15 min (calibration adds ~5 min vs FP16)

Usage:
    python3 scripts/build_tensorrt_engine_int8.py \
        --onnx           models/yolo11n.onnx \
        --calibration-dir data/calibration_frames \
        --output-dir     models/ \
        --workspace-gb   2
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import to_numpy_input

_TRT_MAJOR = int(trt.__version__.split('.')[0])


# ── INT8 Calibrator ──────────────────────────────────────────────────────────

class YOLOInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """Feed letterboxed calibration frames to TensorRT's INT8 entropy calibrator.

    Uses torch GPU tensors for the calibration buffer — no pycuda needed.
    Calibration cache is written after the first run and reused on subsequent
    builds (engine rebuild without re-calibration takes ~2 min instead of ~15).
    """

    def __init__(self, image_dir: str, input_shape: tuple, batch_size: int, cache_file: str):
        super().__init__()
        self.batch_size  = batch_size
        self.input_shape = input_shape          # (C, H, W)
        self.cache_file  = cache_file
        self.idx         = 0

        self.images = sorted(
            glob.glob(os.path.join(image_dir, '*.jpg')) +
            glob.glob(os.path.join(image_dir, '*.png'))
        )
        if not self.images:
            raise FileNotFoundError(
                f'No calibration images found in {image_dir}\n'
                f'Run scripts/extract_calibration_frames.py first.'
            )
        print(f'Calibrator  : {len(self.images)} images  (batch_size={batch_size})')

        # Persistent GPU buffer — TRT reads via data_ptr()
        self._gpu_buf = torch.zeros(
            batch_size, *input_shape, dtype=torch.float32, device='cuda'
        ).contiguous()

    # ── trt.IInt8EntropyCalibrator2 interface ────────────────────────────────

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names):
        """Load next batch of images into GPU buffer. Return None when exhausted."""
        if self.idx + self.batch_size > len(self.images):
            return None

        batch_paths = self.images[self.idx : self.idx + self.batch_size]
        batch_np    = np.zeros(
            (self.batch_size, *self.input_shape), dtype=np.float32
        )

        for i, path in enumerate(batch_paths):
            img = cv2.imread(path)
            if img is None:
                raise RuntimeError(f'Failed to read calibration image: {path}')
            batch_np[i] = to_numpy_input(img)[0]   # (C, H, W) float32

        self._gpu_buf.copy_(torch.from_numpy(batch_np))
        self.idx += self.batch_size

        if self.idx % (self.batch_size * 10) == 0 or self.idx >= len(self.images):
            print(f'  Calibration progress: {min(self.idx, len(self.images))}/{len(self.images)} images')

        return [self._gpu_buf.data_ptr()]

    def read_calibration_cache(self):
        """Return cached calibration data if available, else None (triggers re-calibration)."""
        if os.path.exists(self.cache_file):
            print(f'[INFO] Loading calibration cache: {self.cache_file}')
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Persist calibration data so future builds skip re-calibration."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f'[INFO] Calibration cache saved: {self.cache_file}')


# ── Engine builder ────────────────────────────────────────────────────────────

def build_int8_engine(
    onnx_path:       str,
    engine_path:     str,
    calibration_dir: str,
    cache_file:      str,
    workspace_gb:    int,
    batch_size:      int,
    input_size:      int,
) -> str:
    print(f'\n── Building INT8 engine ─────────────────────────────────────')
    print(f'TRT version       : {trt.__version__}')
    print(f'ONNX source       : {onnx_path}')
    print(f'Output            : {engine_path}')
    print(f'Calibration dir   : {calibration_dir}')
    print(f'Calibration cache : {cache_file}')
    print(f'Workspace         : {workspace_gb} GB\n')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for INT8 calibration.')

    TRT_LOGGER  = trt.Logger(trt.Logger.WARNING)
    calibrator  = YOLOInt8Calibrator(
        image_dir    = calibration_dir,
        input_shape  = (3, input_size, input_size),
        batch_size   = batch_size,
        cache_file   = cache_file,
    )

    with trt.Builder(TRT_LOGGER) as builder:
        if _TRT_MAJOR >= 10:
            network = builder.create_network()
        else:
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )

        with network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
            )

            # Enable INT8 — FP16 must also be enabled as INT8 fallback
            config.set_flag(trt.BuilderFlag.INT8)
            config.set_flag(trt.BuilderFlag.FP16)
            config.int8_calibrator = calibrator

            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f'[ONNX Error {i}] {parser.get_error(i)}')
                    raise RuntimeError('ONNX parsing failed.')

            print('Running calibration + engine build (this takes ~10–15 min on T4) ...')
            t0         = time.perf_counter()
            serialized = builder.build_serialized_network(network, config)
            elapsed    = time.perf_counter() - t0

            if serialized is None:
                raise RuntimeError('INT8 engine build returned None — check logs above.')

            os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(serialized)

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f'\n[DONE] INT8 engine → {engine_path}  ({size_mb:.1f} MB)  in {elapsed:.0f}s')
    print('Tip: copy the .engine and .cache to Google Drive to reuse across sessions.')
    return engine_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Batch 03: Build TRT INT8 engine with calibration')
    p.add_argument('--onnx',            default='models/yolo11n.onnx')
    p.add_argument('--calibration-dir', default='data/calibration_frames',
                   help='Directory of letterboxed JPEG frames (from extract_calibration_frames.py)')
    p.add_argument('--output-dir',      default='models/')
    p.add_argument('--cache-file',      default='models/calibration.cache',
                   help='Path to save/load calibration cache — reuse avoids re-calibration')
    p.add_argument('--input-size',      type=int, default=640)
    p.add_argument('--batch-size',      type=int, default=1,
                   help='Calibration batch size (keep at 1 to match inference batch size)')
    p.add_argument('--workspace-gb',    type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in [(args.onnx, 'ONNX model'), (args.calibration_dir, 'calibration dir')]:
        if not os.path.exists(path):
            print(f'[ERROR] {label} not found: {path}')
            sys.exit(1)

    base       = os.path.splitext(os.path.basename(args.onnx))[0]
    engine_path = os.path.join(args.output_dir, f'{base}_int8.engine')

    build_int8_engine(
        onnx_path       = args.onnx,
        engine_path     = engine_path,
        calibration_dir = args.calibration_dir,
        cache_file      = args.cache_file,
        workspace_gb    = args.workspace_gb,
        batch_size      = args.batch_size,
        input_size      = args.input_size,
    )


if __name__ == '__main__':
    main()

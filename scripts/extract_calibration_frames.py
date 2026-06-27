"""Batch 03: Extract evenly-spaced frames from video for TRT INT8 calibration.

Saves letterboxed 640x640 JPEGs into data/calibration_frames/.
500 frames is the recommended minimum for good INT8 calibration quality.
Run this once per video — output is reusable across Colab sessions if saved to Drive.

Usage:
    python3 scripts/extract_calibration_frames.py \
        --video  data/clip.mp4 \
        --output data/calibration_frames \
        --frames 500 \
        --size   640
"""

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import letterbox


def parse_args():
    p = argparse.ArgumentParser(description='Extract calibration frames for TRT INT8')
    p.add_argument('--video',  default='data/clip.mp4')
    p.add_argument('--output', default='data/calibration_frames',
                   help='Directory to save letterboxed JPEG frames')
    p.add_argument('--frames', type=int, default=500,
                   help='Number of frames to extract (evenly spaced)')
    p.add_argument('--size',   type=int, default=640,
                   help='Square input size to letterbox to')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open video: {args.video}')
        sys.exit(1)

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n      = min(args.frames, total)

    print(f'Video   : {args.video}')
    print(f'Source  : {width}x{height} @ {fps:.1f} FPS  ({total} frames)')
    print(f'Target  : {n} calibration frames → {args.output}')
    print(f'Size    : {args.size}x{args.size} (letterboxed)\n')

    # Evenly spaced frame indices across the full video
    indices = set(int(i * total / n) for i in range(n))

    saved   = 0
    frame_i = 0
    input_shape = (args.size, args.size)

    while saved < n:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i in indices:
            lb, _, _ = letterbox(frame, input_shape)
            out_path = os.path.join(args.output, f'frame_{frame_i:06d}.jpg')
            cv2.imwrite(out_path, lb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            if saved % 50 == 0 or saved == n:
                print(f'  Saved {saved}/{n} frames ...')
        frame_i += 1

    cap.release()

    actual_size_mb = sum(
        os.path.getsize(os.path.join(args.output, f))
        for f in os.listdir(args.output)
        if f.endswith('.jpg')
    ) / 1e6

    print(f'\nDone — {saved} frames saved to {args.output}  ({actual_size_mb:.1f} MB)')
    print('Tip: copy this folder to Google Drive to avoid re-extracting each Colab session.')


if __name__ == '__main__':
    main()

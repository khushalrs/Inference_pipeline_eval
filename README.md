# Real-Time Driving Perception Inference Optimization

Build and benchmark a real-time autonomous-driving-style perception pipeline using pretrained YOLO11n, then optimize inference latency, throughput, and bottlenecks across PyTorch, ONNX Runtime, and TensorRT.

## Project Goals

- Export and deploy pretrained models across runtimes
- Profile preprocessing, model execution, and postprocessing
- Compare FP32 vs FP16 inference
- Measure latency percentiles (p50, p90, p99) and end-to-end FPS
- Document bottleneck-driven engineering decisions

## Repository Structure

```
├── notebooks/          # Runner notebook for Colab execution
├── scripts/            # Entrypoint scripts (called via !python3 in Colab)
├── src/                # Shared utility modules
├── data/               # Input video (not tracked in git — see below)
├── results/            # Benchmark CSVs (not tracked in git)
├── plots/              # Output charts (not tracked in git)
└── reports/            # Final written report
```

## Input Video

Primary benchmark video: Mixkit "Dashboard of a car" (1920x1080, 18s, 24 FPS).  
Download manually and place at `data/clip.mp4`, or mount from Google Drive in the runner notebook.

## Setup (Colab)

All scripts are designed to run on Google Colab with a GPU runtime.

```bash
git clone <repo-url>
cd real-time-driving-inference-optimizer
pip install -r requirements.txt
```

Then run phases using the runner notebook in `notebooks/`.

## Phases

| Phase | Description |
|-------|-------------|
| 1 | Baseline PyTorch inference pipeline |
| 2 | Detailed benchmarking (latency percentiles, stage breakdown) |
| 3 | ONNX export and validation |
| 4 | TensorRT FP32 and FP16 optimization |
| 5 | Pipeline bottleneck optimization |
| 6 | Final report and portfolio packaging |

## Model

**YOLO11n** (Ultralytics) — pretrained on COCO, no training required.

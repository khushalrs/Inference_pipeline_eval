# Batch 01 — Naive Baseline

## Setup
- Hardware: Google Colab T4 GPU
- Model: YOLO11n (Ultralytics, pretrained COCO)
- Input: Dashcam clip (1920×1080, 24 FPS), looped to 2516 frames
- Runtimes: PyTorch eager, ONNX Runtime (CPU EP), TRT FP32, TRT FP16

## What was measured
End-to-end pipeline including: disk read → preprocess → inference → postprocess.
All four stages timed together in a single loop per frame.

## Key results
| Runtime     | Inference p50 (ms) | Total p50 (ms) | FPS  |
|-------------|-------------------|----------------|------|
| PyTorch     | 18.5              | 74.6           | 12.7 |
| ONNX        | 13.0              | 68.1           | 13.2 |
| TRT FP32    | 8.2               | 68.5           | 13.7 |
| TRT FP16    | 8.1               | 68.7           | 13.8 |

## Known Issues (fixed in Batch 02)

**1. Warmup contamination**
Frame 0 cold-start artifacts inflate p99/max across all runtimes.
- TRT FP32 postprocess p99: 546ms (vs ~3ms steady-state)
- TRT FP16 max total: 402ms
- PyTorch max total: 761ms
No warmup frames were excluded before computing statistics.

**2. I/O mixed into inference timing**
The `read` stage (disk → frame decode) averages 52–56ms per frame, accounting
for ~70% of total latency. This drowns out inference differences across runtimes.
All four runtimes show near-identical total FPS (12.7–13.8) despite 2.4× inference
speedup from PyTorch (20ms) to TRT FP16 (8ms).
Methodology does not match MLPerf or NVIDIA trtexec standards, which isolate
compute from data delivery.

**3. ONNX running on CPU Execution Provider**
ONNX inference at 14ms vs TRT at 8ms is partly a CPU vs GPU comparison.
Should be re-run with CUDAExecutionProvider for a valid comparison.

**4. Missing TRT FP16 raw timings**
Per-frame CSV was not saved for TRT FP16. Only aggregated benchmark stats exist.
Cannot generate per-frame latency timeline for FP16.

**5. torch.compile not included**
No intermediate runtime between PyTorch eager and TRT. Optimization ladder is incomplete.

## What this batch is good for
- Establishing a true naive baseline before any fixes
- Demonstrating that I/O dominates when not isolated
- Showing that runtime optimizations are invisible when methodology is flawed
- The conf_sweep experiment (batch_01b) is the most methodologically clean result
  in this batch — it measures inference+postprocess only (I/O already excluded there)

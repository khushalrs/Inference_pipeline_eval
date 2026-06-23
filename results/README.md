# Results

Benchmark CSVs are not tracked in git. Download manually from Colab after each phase.

## Expected Files

| File | Phase | Description |
|------|-------|-------------|
| `pytorch_baseline.csv` | 2 | Per-frame latency breakdown for PyTorch |
| `onnx_benchmark.csv` | 3 | Per-frame latency for ONNX Runtime |
| `tensorrt_fp32.csv` | 4 | Per-frame latency for TensorRT FP32 |
| `tensorrt_fp16.csv` | 4 | Per-frame latency for TensorRT FP16 |
| `optimization_comparison.csv` | 5 | Before/after optimization results |

# Project Scope: Real-Time Driving Perception Inference Optimization

## Project Title

**Real-Time Driving Perception Inference Optimization using PyTorch, ONNX, and TensorRT**

## One-Line Summary

Build and benchmark a real-time autonomous-driving-style perception pipeline using pretrained vision models, then optimize inference latency, throughput, and bottlenecks across different runtimes and precisions.

## Goal

The goal is to demonstrate practical inference engineering knowledge by moving beyond model training and focusing on:

- model export and runtime deployment
- FP32 vs FP16 inference
- preprocessing, model execution, and postprocessing profiling
- latency and throughput tradeoffs
- output consistency after optimization
- bottleneck-driven engineering decisions

This project is intended to support applications for AI inference, autonomy, robotics, and systems-oriented software engineering roles.

---

## Primary Input Video

For the first version, use a short public driving clip so the pipeline can be built quickly without dataset setup friction.

**Primary video:** Mixkit - "Dashboard of a car"  
**Link:** https://mixkit.co/free-stock-video/dashboard-of-a-car-72/

Notes:

- Duration: 18 seconds
- Frame rate: 24 FPS
- Available resolutions: 1080x720, 1920x1080, 4096x2160
- Use the 1920x1080 version for the main benchmark
- Optionally downscale to 640x640 or 1280x720 before inference depending on model input size

### Optional Dataset-Style Inputs

Use these only if more realistic driving data is needed later:

- BDD100K: https://github.com/bdd100k/bdd100k
- BDD100K download page: https://bdd-data.berkeley.edu/download.html
- KITTI Raw Dataset: https://www.cvlibs.net/datasets/kitti/raw_data.php
- Pixabay dashcam videos: https://pixabay.com/videos/search/car%20dashcam/
- Pexels dashcam videos: https://www.pexels.com/search/videos/dashcam/

---

## Models

### Primary Model

**YOLOv8n / YOLO11n / latest Ultralytics nano detector**

Reason:

- pretrained weights are available
- easy PyTorch-to-ONNX export
- supports TensorRT export workflows
- suitable for real-time object detection
- strong fit for autonomous perception-style benchmarking

### Optional Secondary Model

**YOLOP or another lightweight driving-perception model**

Reason:

- more autonomous-driving-specific
- can add drivable-area/lane-related perception
- useful for a second phase if time allows

### No Training Required

This project will use pretrained models only. The focus is inference optimization, not model accuracy improvement.

---

## Project Phases

## Phase 1: Repository Setup and Baseline Pipeline

### Objective

Create a clean, reproducible project structure and run a pretrained detection model on the selected driving video.

### Tasks

- Set up project repository
- Add environment setup instructions
- Download/link the selected driving video
- Load video frame-by-frame
- Run pretrained YOLO inference in PyTorch
- Save annotated output video
- Save raw timing logs

### Deliverables

- Working PyTorch inference script
- Input video link documented
- Annotated output video
- Initial baseline FPS and latency numbers

### Success Criteria

The pipeline can process the full video and produce an annotated output video with basic timing measurements.

---

## Phase 2: Baseline Benchmarking

### Objective

Measure baseline PyTorch performance in a reliable way.

### Tasks

- Add warmup iterations
- Measure per-frame latency
- Record average, p50, p90, and p99 latency
- Separate timing into:
  - video read/decode
  - preprocessing
  - model inference
  - postprocessing/NMS
  - visualization/output writing
- Log FPS and GPU/CPU memory where possible

### Deliverables

- `results/pytorch_baseline.csv`
- Baseline latency table
- First pipeline breakdown chart

### Success Criteria

The project reports end-to-end latency and individual pipeline-stage latency, not just model forward-pass time.

---

## Phase 3: ONNX Export and Validation

### Objective

Export the pretrained model to ONNX and verify that the exported model produces consistent outputs.

### Tasks

- Export PyTorch model to ONNX
- Run ONNX Runtime inference
- Compare PyTorch and ONNX outputs
- Record output drift using simple checks such as:
  - number of detections
  - class consistency
  - confidence score difference
  - bounding box coordinate difference
- Benchmark ONNX Runtime latency

### Deliverables

- ONNX model file
- ONNX inference script
- PyTorch vs ONNX validation notes
- `results/onnx_benchmark.csv`

### Success Criteria

ONNX inference runs successfully and output differences are documented rather than ignored.

---

## Phase 4: TensorRT / NVIDIA GPU Optimization

### Objective

Build and benchmark TensorRT engines on a free NVIDIA GPU environment such as Kaggle or Google Colab.

### Tasks

- Set up NVIDIA GPU notebook/runtime
- Export/build TensorRT engine
- Benchmark FP32 TensorRT
- Benchmark FP16 TensorRT
- Compare against PyTorch and ONNX baselines
- Record GPU name, CUDA version, TensorRT version, and batch size

### Deliverables

- TensorRT benchmark notebook/script
- `results/tensorrt_fp32.csv`
- `results/tensorrt_fp16.csv`
- Runtime comparison table

### Success Criteria

The project demonstrates whether TensorRT FP16 improves model inference latency and how much of that improvement appears in end-to-end FPS.

---

## Phase 5: Pipeline Bottleneck Optimization

### Objective

Apply small but meaningful engineering optimizations based on profiling results.

### Candidate Optimizations

Choose 2-3 depending on time:

1. **Batch-size sweep**
   - Test batch sizes 1, 2, 4, and 8
   - Compare real-time latency vs throughput

2. **Resolution sweep**
   - Test 640x640, 960x540, and 1280x720 style inputs
   - Show accuracy/visual quality vs speed tradeoff

3. **Postprocessing/NMS tuning**
   - Tune confidence threshold
   - Add top-k filtering before NMS
   - Measure postprocessing reduction

4. **Preprocessing improvement**
   - Avoid repeated unnecessary conversions
   - Minimize CPU-GPU transfer overhead
   - Use batched tensor preprocessing where useful

5. **Async or staged pipeline**
   - Separate video loading, inference, and writing where possible
   - Measure FPS improvement

### Deliverables

- Before/after benchmark tables
- One or two optimization charts
- Short explanation of what changed and why

### Success Criteria

The final report includes at least one real engineering insight such as:

> Model execution improved significantly with TensorRT FP16, but end-to-end speedup was limited because preprocessing and postprocessing became a larger share of total runtime.

---

## Phase 6: Final Report and Portfolio Packaging

### Objective

Package the work so it clearly communicates inference engineering knowledge.

### Final Report Sections

1. Project overview
2. Hardware/software setup
3. Input video and model selection
4. Inference pipeline design
5. Benchmark methodology
6. Runtime comparison: PyTorch vs ONNX vs TensorRT
7. FP32 vs FP16 results
8. Batch-size and resolution tradeoffs
9. Pipeline bottleneck analysis
10. Key insights and limitations
11. Resume bullets
12. Future work

### Required Figures/Tables

- Latency comparison table
- p50/p90/p99 latency chart
- End-to-end FPS comparison
- Pipeline breakdown chart
- Before/after optimization table

### Deliverables

- `reports/final_inference_optimization_report.md`
- `results/*.csv`
- `plots/*.png`
- README with reproduction steps
- optional short demo GIF or annotated video

### Success Criteria

A reviewer should be able to understand:

- what was optimized
- how it was measured
- what bottleneck appeared
- what engineering decision was made
- what improved and what did not

---

## Suggested Repository Structure

```text
real-time-driving-inference-optimizer/
├── README.md
├── requirements.txt
├── notebooks/
│   └── tensorrt_colab_or_kaggle.ipynb
├── scripts/
│   ├── run_pytorch_video.py
│   ├── export_onnx.py
│   ├── run_onnx_video.py
│   ├── build_tensorrt_engine.py
│   ├── benchmark.py
│   └── plot_results.py
├── src/
│   ├── video_io.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   ├── timing.py
│   └── metrics.py
├── data/
│   └── README.md
├── results/
│   └── README.md
├── plots/
│   └── README.md
└── reports/
    └── final_inference_optimization_report.md
```

---

## Final Project Outcome

By the end, this should not read like a generic benchmark. It should read like a small inference engineering case study.

Expected final claim format:

> Built and optimized a real-time driving perception inference pipeline using pretrained YOLO models, PyTorch, ONNX Runtime, and TensorRT. Profiled preprocessing, model execution, and postprocessing bottlenecks; compared FP32/FP16 precision and batch-size tradeoffs; improved end-to-end FPS by Xx while documenting accuracy/output drift and latency percentiles.

---

## Stretch Goals

Add these only after the main pipeline works:

- INT8 post-training quantization
- TensorRT dynamic shapes
- GPU-based NMS
- CoreML/MPS comparison on MacBook
- second video with night/rain/traffic conditions
- YOLOP/HybridNets-style multitask perception
- Dockerfile for reproducibility

---

## What This Project Should Prove

This project should demonstrate that you understand inference beyond surface-level model usage:

- runtime choice matters
- precision affects speed and correctness
- benchmarking needs synchronization and warmup
- average latency is not enough
- postprocessing can dominate after model optimization
- end-to-end FPS is often limited by non-model code
- optimization decisions should be based on measurements, not assumptions

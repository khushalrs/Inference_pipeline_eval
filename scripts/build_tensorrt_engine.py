"""Phase 4: Build TensorRT FP32 and FP16 engines from the exported ONNX model.

TensorRT is pre-installed on Colab GPU runtimes — no extra pip install needed.
Engine build is hardware-specific: an engine built on a T4 will not run on a
different GPU, and cannot be reused across sessions if the TRT version changes.
Rebuild at the start of every fresh Colab session.

Build times on T4 GPU:  FP32 ~ 2–5 min  /  FP16 ~ 3–7 min

Usage (build both precisions in one call):
    python3 scripts/build_tensorrt_engine.py \
        --onnx         models/yolo11n.onnx \
        --output-dir   models/ \
        --fp32 --fp16 \
        --workspace-gb 2
"""

import argparse
import os
import sys
import time

import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TRT_MAJOR = int(trt.__version__.split('.')[0])


def parse_args():
    p = argparse.ArgumentParser(description='Phase 4: Build TRT FP32 / FP16 engines')
    p.add_argument('--onnx',         default='models/yolo11n.onnx',
                   help='Source ONNX model (from export_onnx.py)')
    p.add_argument('--output-dir',   default='models/',
                   help='Directory to save .engine files')
    p.add_argument('--fp32',         action='store_true', help='Build FP32 engine')
    p.add_argument('--fp16',         action='store_true', help='Build FP16 engine')
    p.add_argument('--workspace-gb', type=int, default=2,
                   help='TRT builder max workspace in GB')
    return p.parse_args()


def _engine_path(output_dir: str, onnx_path: str, fp16: bool) -> str:
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    prec = 'fp16' if fp16 else 'fp32'
    return os.path.join(output_dir, f'{base}_{prec}.engine')


def build_engine(onnx_path: str, engine_path: str, fp16: bool, workspace_gb: int) -> str:
    prec  = 'FP16' if fp16 else 'FP32'
    print(f'\n── Building {prec} engine ────────────────────────────────')
    print(f'TRT version  : {trt.__version__}')
    print(f'ONNX source  : {onnx_path}')
    print(f'Output       : {engine_path}')
    print(f'Workspace    : {workspace_gb} GB')
    print(f'(First build may take several minutes — subsequent builds use cache)\n')

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder:
        # TRT 10.x: EXPLICIT_BATCH is default; flag is deprecated
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

            if fp16:
                if _TRT_MAJOR < 10 and not builder.platform_has_fast_fp16:
                    print('[WARNING] GPU does not report native FP16 — speed gain may be limited')
                fp16_flag = getattr(trt.BuilderFlag, 'FP16', None)
                if fp16_flag is not None:
                    config.set_flag(fp16_flag)
                else:
                    print('[INFO] TRT 11+: BuilderFlag.FP16 removed; '
                          'TRT will auto-use FP16 Tensor Cores on supported hardware.')

            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        print(f'[ONNX Parse Error {i}] {parser.get_error(i)}')
                    raise RuntimeError(
                        'ONNX parsing failed. Re-export with export_onnx.py and retry.'
                    )

            t0         = time.perf_counter()
            serialized = builder.build_serialized_network(network, config)
            elapsed    = time.perf_counter() - t0

            if serialized is None:
                raise RuntimeError(f'TRT {prec} engine build returned None — check logs above')

            os.makedirs(os.path.dirname(os.path.abspath(engine_path)), exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(serialized)

    size_mb = os.path.getsize(engine_path) / 1e6
    print(f'[DONE] {prec} engine  →  {engine_path}  ({size_mb:.1f} MB)  in {elapsed:.0f}s')
    return engine_path


def main():
    args = parse_args()

    if not os.path.exists(args.onnx):
        print(f'[ERROR] ONNX model not found: {args.onnx}')
        print('        Run scripts/export_onnx.py first.')
        sys.exit(1)

    if not args.fp32 and not args.fp16:
        print('[ERROR] Specify at least --fp32 or --fp16 (or both).')
        sys.exit(1)

    built = []
    if args.fp32:
        path = _engine_path(args.output_dir, args.onnx, fp16=False)
        built.append(build_engine(args.onnx, path, fp16=False, workspace_gb=args.workspace_gb))

    if args.fp16:
        path = _engine_path(args.output_dir, args.onnx, fp16=True)
        built.append(build_engine(args.onnx, path, fp16=True, workspace_gb=args.workspace_gb))

    print(f'\nEngines ready:')
    for p in built:
        print(f'  {p}  ({os.path.getsize(p)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()

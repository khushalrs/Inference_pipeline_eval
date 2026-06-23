"""TensorRT inference runner using PyTorch CUDA tensors as GPU memory.

No pycuda required — input/output buffers are torch.Tensor objects and
execution is submitted onto torch.cuda.current_stream() so CUDATimer's
synchronize() call correctly waits for TRT work to finish.

Supports TensorRT 8.x and 10.x APIs transparently.
"""

import numpy as np
import torch
import tensorrt as trt

_TRT_MAJOR = int(trt.__version__.split('.')[0])


class TRTRunner:
    """Load a serialised TRT engine and run single-batch inference.

    Works with FP32 and FP16 engines — output is always returned as float32
    so the downstream NMS code is identical for both precision modes.
    """

    def __init__(self, engine_path: str, device: str = 'cuda'):
        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._setup_buffers()

    def _setup_buffers(self):
        if _TRT_MAJOR >= 10:
            self._setup_v10()
        else:
            self._setup_v8()

    def _setup_v10(self):
        """TensorRT 10.x buffer setup."""
        for i in range(self.engine.num_io_tensors):
            name   = self.engine.get_tensor_name(i)
            shape  = tuple(self.engine.get_tensor_shape(name))
            np_dt  = trt.nptype(self.engine.get_tensor_dtype(name))
            t_dt   = torch.float16 if np_dt == np.float16 else torch.float32
            buf    = torch.zeros(shape, dtype=t_dt, device=self.device).contiguous()

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_shape = shape
                self._in_buf     = buf
                self._in_name    = name
            else:
                self.output_shape = shape
                self._out_buf     = buf
                self._out_name    = name

        self.context.set_tensor_address(self._in_name,  self._in_buf.data_ptr())
        self.context.set_tensor_address(self._out_name, self._out_buf.data_ptr())

    def _setup_v8(self):
        """TensorRT 8.x buffer setup."""
        self._bindings = []
        for i in range(self.engine.num_bindings):
            shape  = tuple(self.engine.get_binding_shape(i))
            np_dt  = trt.nptype(self.engine.get_binding_dtype(i))
            t_dt   = torch.float16 if np_dt == np.float16 else torch.float32
            buf    = torch.zeros(shape, dtype=t_dt, device=self.device).contiguous()
            self._bindings.append(buf.data_ptr())

            if self.engine.binding_is_input(i):
                self.input_shape = shape
                self._in_buf     = buf
            else:
                self.output_shape = shape
                self._out_buf     = buf

    def infer(self, input_np: np.ndarray) -> torch.Tensor:
        """Run inference on a single float32 numpy input array.

        Copies input to GPU, executes TRT on the current PyTorch CUDA stream,
        and returns the output as a float32 tensor (cast from FP16 if needed).
        The caller must NOT call torch.cuda.synchronize() before reading the
        result — CUDATimer.__exit__ does this correctly.
        """
        # Host → device copy (async on current stream)
        self._in_buf.copy_(torch.from_numpy(input_np).to(self.device),
                           non_blocking=True)

        stream = torch.cuda.current_stream().cuda_stream
        if _TRT_MAJOR >= 10:
            self.context.execute_async_v3(stream)
        else:
            self.context.execute_async_v2(self._bindings, stream)

        # Return float32 so NMS code needs no modification for FP16 engines
        return self._out_buf if self._out_buf.dtype == torch.float32 else self._out_buf.float()

    def __del__(self):
        try:
            del self.context
            del self.engine
        except Exception:
            pass

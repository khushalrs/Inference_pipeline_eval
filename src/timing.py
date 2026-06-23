import time
import torch


class CUDATimer:
    """Context manager for accurate GPU-synchronized wall-clock timing.

    Calls torch.cuda.synchronize() before and after the timed block so
    GPU work queued before the block does not bleed into the measurement.
    Falls back to plain wall-clock timing when CUDA is unavailable.
    """

    def __init__(self):
        self.elapsed_ms = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0

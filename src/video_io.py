import cv2


class VideoReader:
    """Thin wrapper around cv2.VideoCapture exposing metadata and a read() method."""

    def __init__(self, path: str):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        self.fps         = self._cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width       = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        """Return (success, frame). Matches cv2.VideoCapture.read() signature."""
        return self._cap.read()

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


def load_frames(path: str, input_shape: tuple = None) -> tuple:
    """Decode all frames into RAM, optionally letterboxing during buffering.

    Args:
        path:        Video file path.
        input_shape: If given (H, W), each frame is letterboxed to this size
                     before storing. Reduces memory from ~60 GB (raw 4K) to
                     ~3 GB (640×640) — required for Colab when source is 4K.

    Returns:
        frames:  list of (letterboxed BGR uint8 ndarray, pad) if input_shape
                 given, else list of raw BGR uint8 ndarrays.
        fps, width, height: source video metadata.
    """
    import sys

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if input_shape is not None:
        from src.preprocessing import letterbox as _letterbox
        print(f'Pre-buffering {total} frames with letterbox → {input_shape} '
              f'(source {width}x{height} @ {fps:.1f} FPS) ...')
        stored_mb = total * input_shape[0] * input_shape[1] * 3 / 1e6
    else:
        print(f'Pre-buffering {total} frames at full resolution '
              f'({width}x{height} @ {fps:.1f} FPS) ...')
        stored_mb = total * width * height * 3 / 1e6

    print(f'Estimated buffer size: {stored_mb:.0f} MB')

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if input_shape is not None:
            lb, ratio, pad = _letterbox(frame, input_shape)
            frames.append((lb, pad))
        else:
            frames.append(frame)
    cap.release()

    print(f'Pre-buffered {len(frames)}/{total} frames — I/O decoupled from inference loop.\n')
    return frames, fps, width, height


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame):
        self._writer.write(frame)

    def release(self):
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

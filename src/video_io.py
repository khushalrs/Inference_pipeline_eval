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


def load_frames(path: str) -> tuple:
    """Decode all frames from a video file into a list of numpy arrays.

    Returns (frames, fps, width, height) where frames is a list of BGR uint8
    arrays. Call this once before the benchmark loop so disk I/O is excluded
    from inference timing — mirrors how a real camera streams into RAM.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    print(f'Pre-buffered {len(frames)}/{total} frames '
          f'({width}x{height} @ {fps:.1f} FPS) into RAM — I/O decoupled from inference loop.')
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

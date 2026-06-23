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

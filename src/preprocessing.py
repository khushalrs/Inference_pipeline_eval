import cv2
import numpy as np
import torch


def letterbox(img: np.ndarray, new_shape: tuple = (640, 640)):
    """Resize image to new_shape with letterbox padding (grey fill = 114).

    Returns:
        img_padded: resized and padded image (new_shape HxW)
        ratio:      scale factor applied to original dimensions
        pad:        (left_pad, top_pad) in pixels — needed to undo during box scaling
    """
    h, w = img.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape[1] - new_w
    pad_h = new_shape[0] - new_h
    left, top     = pad_w // 2, pad_h // 2
    right, bottom = pad_w - left, pad_h - top

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return img_padded, ratio, (left, top)


def to_tensor(img_bgr: np.ndarray, device: str = 'cuda') -> torch.Tensor:
    """Convert a BGR HxWxC uint8 numpy array to a normalised 1xCxHxW float tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_chw = np.ascontiguousarray(img_rgb.transpose(2, 0, 1))
    tensor  = torch.from_numpy(img_chw).float().div_(255.0)
    return tensor.unsqueeze(0).to(device)

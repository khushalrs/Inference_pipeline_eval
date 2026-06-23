import cv2
import torch
from ultralytics.utils.ops import non_max_suppression


def run_nms(
    raw_preds,
    conf_thres: float = 0.25,
    iou_thres:  float = 0.45,
):
    """Wrap ultralytics NMS. Returns a list of (N, 6) tensors — one per batch item."""
    return non_max_suppression(raw_preds, conf_thres=conf_thres, iou_thres=iou_thres)


def scale_boxes(
    boxes:       torch.Tensor,
    orig_shape:  tuple,
    input_shape: tuple = (640, 640),
    pad:         tuple = (0, 0),
) -> torch.Tensor:
    """Scale boxes from letterboxed input coordinates back to original frame coordinates.

    Args:
        boxes:       (N, 4) xyxy tensor in letterboxed-input space
        orig_shape:  (height, width) of the original video frame
        input_shape: (height, width) of the model input (e.g. 640x640)
        pad:         (left_pad, top_pad) returned by preprocessing.letterbox()
    """
    gain   = min(input_shape[0] / orig_shape[0], input_shape[1] / orig_shape[1])
    result = boxes.clone().float()
    result[:, [0, 2]] -= pad[0]   # remove left pad from x coords
    result[:, [1, 3]] -= pad[1]   # remove top  pad from y coords
    result /= gain
    result[:, [0, 2]] = result[:, [0, 2]].clamp(0, orig_shape[1])  # x within width
    result[:, [1, 3]] = result[:, [1, 3]].clamp(0, orig_shape[0])  # y within height
    return result


def draw_detections(frame, detections, class_names: dict):
    """Draw bounding boxes and confidence labels onto frame in-place.

    Args:
        frame:       BGR numpy array (original resolution)
        detections:  (N, 6) tensor  [x1, y1, x2, y2, conf, cls]  or None/empty
        class_names: dict {int -> str} from YOLO model
    """
    if detections is None or len(detections) == 0:
        return frame

    for *xyxy, conf, cls_id in detections.tolist():
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        label = f"{class_names.get(int(cls_id), str(int(cls_id)))} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return frame

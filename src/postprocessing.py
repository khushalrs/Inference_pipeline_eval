import cv2
import torch
import torchvision


def _xywh2xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2)."""
    xy = boxes[:, :2]
    wh = boxes[:, 2:]
    return torch.cat([xy - wh / 2, xy + wh / 2], dim=1)


def run_nms(
    raw_preds,
    conf_thres: float = 0.25,
    iou_thres:  float = 0.45,
    max_det:    int   = 300,
):
    """NMS for YOLO11n raw predictions using torchvision.ops.batched_nms.

    Args:
        raw_preds:  (batch, 4+nc, num_anchors) tensor from model forward pass.
                    Boxes are in (cx, cy, w, h) pixel-space; class scores are sigmoid-activated.
        conf_thres: minimum class confidence to keep a box.
        iou_thres:  IoU suppression threshold.
        max_det:    maximum detections returned per image.

    Returns:
        List of (N, 6) tensors [x1, y1, x2, y2, conf, cls], one per batch item.
    """
    if raw_preds.ndim == 3:
        batch = [raw_preds[i] for i in range(raw_preds.shape[0])]
    else:
        batch = [raw_preds]

    results = []
    for pred in batch:
        # pred: (4+nc, num_anchors) → (num_anchors, 4+nc)
        pred         = pred.T
        boxes_xywh   = pred[:, :4]
        class_scores = pred[:, 4:]

        conf, cls = class_scores.max(dim=1)

        keep_mask = conf >= conf_thres
        if keep_mask.sum() == 0:
            results.append(torch.zeros((0, 6), device=raw_preds.device))
            continue

        boxes_xywh = boxes_xywh[keep_mask]
        conf       = conf[keep_mask]
        cls        = cls[keep_mask]

        boxes_xyxy = _xywh2xyxy(boxes_xywh)

        keep = torchvision.ops.batched_nms(
            boxes_xyxy.float(), conf.float(), cls, iou_thres
        )[:max_det]

        det = torch.cat(
            [boxes_xyxy[keep], conf[keep].unsqueeze(1), cls[keep].float().unsqueeze(1)],
            dim=1,
        )
        results.append(det)

    return results


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

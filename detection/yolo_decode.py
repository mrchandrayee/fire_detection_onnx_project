from __future__ import annotations
from typing import Sequence
import cv2
import numpy as np
from .schema import Det

COCO_NAMES = {
    0:"person", 1:"bicycle", 2:"car", 3:"motorcycle", 4:"airplane",
    5:"bus", 6:"train", 7:"truck", 8:"boat", 9:"traffic light",
    10:"fire hydrant", 11:"stop sign", 12:"parking meter", 13:"bench",
    14:"bird", 15:"cat", 16:"dog", 17:"horse", 18:"sheep", 19:"cow",
    20:"elephant", 21:"bear", 22:"zebra", 23:"giraffe"
}

def letterbox(frame: np.ndarray, size: int = 640):
    """Resize with aspect ratio preserved; returns NCHW RGB float tensor and transform info."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor, scale, float(pad_x), float(pad_y)

def _nms(boxes, scores, iou_threshold=0.45):
    if not boxes:
        return []
    xywh = []
    for x1,y1,x2,y2 in boxes:
        xywh.append([x1,y1,x2-x1,y2-y1])
    return cv2.dnn.NMSBoxes(xywh, scores, score_threshold=0.0,
                            nms_threshold=iou_threshold)

def decode_yolo_output(output: np.ndarray, scale: float, pad_x: float, pad_y: float,
                       confidence: float = 0.35, names: Sequence[str] | None = None,
                       iou_threshold: float = 0.45, frame_shape=None) -> list[Det]:
    """Decode common YOLOv8 [1,4+classes,N]/[1,N,4+classes] and YOLOv5 [N,5+classes] outputs."""
    original = np.asarray(output)
    arr = np.squeeze(original)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported ONNX output shape: {original.shape}")
    # YOLOv8 commonly has channels first (84 x 8400); transpose to rows.
    if arr.shape[0] != 1 and arr.shape[0] <= 256 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] < 5:
        raise ValueError(f"Expected at least 5 output columns, got {arr.shape}")

    boxes, scores, class_ids = [], [], []
    for row in arr:
        if row.shape[0] == 5:
            cls_id, score = 0, float(row[4])
            cx, cy, bw, bh = map(float, row[:4])
        elif row.shape[0] > 5:
            # YOLOv5 export includes objectness; YOLOv8 does not.
            if row.shape[0] == 6:  # end-to-end NMS export: x1,y1,x2,y2,score,class
                x1,y1,x2,y2,score,cls = map(float,row)
                if score < confidence: continue
                boxes.append([x1,y1,x2,y2]); scores.append(score); class_ids.append(int(cls))
                continue
            if row.shape[0] >= 85:
                obj = float(row[4])
                cls_scores = row[5:]
                cls_id = int(np.argmax(cls_scores))
                score = obj * float(cls_scores[cls_id])
            else:
                cls_scores = row[4:]
                cls_id = int(np.argmax(cls_scores))
                score = float(cls_scores[cls_id])
            cx, cy, bw, bh = map(float, row[:4])
        else:
            continue
        if score < confidence: continue
        boxes.append([cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2])
        scores.append(score); class_ids.append(cls_id)

    if not boxes:
        return []
    keep = _nms(boxes, scores, iou_threshold)
    if hasattr(keep, "flatten"):
        keep = keep.flatten().tolist()
    elif keep is None:
        keep = []
    h, w = frame_shape[:2] if frame_shape is not None else (None, None)
    results = []
    for i in keep:
        x1,y1,x2,y2 = boxes[int(i)]
        x1=(x1-pad_x)/scale; x2=(x2-pad_x)/scale
        y1=(y1-pad_y)/scale; y2=(y2-pad_y)/scale
        if w is not None:
            x1=max(0,min(w-1,x1)); x2=max(0,min(w-1,x2))
            y1=max(0,min(h-1,y1)); y2=max(0,min(h-1,y2))
        cid=class_ids[int(i)]
        if names and 0 <= cid < len(names): label=names[cid]
        else: label=COCO_NAMES.get(cid, f"class_{cid}")
        results.append(Det((x1,y1,x2,y2),scores[int(i)],cid,label))
    return results

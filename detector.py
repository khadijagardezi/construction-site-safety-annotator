from __future__ import annotations

import os
import urllib.request
from typing import Dict, List

import numpy as np

WEIGHTS_PATH = "ppe_yolov8.pt"
WEIGHTS_URL = "https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt"

_OBJECT_MAP = {"Person": "person", "machinery": "heavy_vehicle", "vehicle": "vehicle"}

_HARDHAT, _NO_HARDHAT = "Hardhat", "NO-Hardhat"
_VEST, _NO_VEST = "Safety Vest", "NO-Safety Vest"

_PERSON_COLOR = (0, 200, 0)
_VEHICLE_COLOR = (255, 140, 0)
_ALERT_COLOR = (220, 40, 40)


def load_model(weights: str = WEIGHTS_PATH):
    from ultralytics import YOLO

    if not os.path.exists(weights):
        urllib.request.urlretrieve(WEIGHTS_URL, weights)
    return YOLO(weights)


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def detect(model, image, conf: float = 0.25) -> List[Dict]:
    r = model(image, conf=conf, verbose=False)[0]

    objects: List[Dict] = []
    signals: List[Dict] = []
    for box in r.boxes:
        name = model.names[int(box.cls[0])]
        c = float(box.conf[0])
        xyxy = [int(v) for v in box.xyxy[0].tolist()]
        if name in _OBJECT_MAP:
            objects.append(
                {"id": len(objects), "type": _OBJECT_MAP[name],
                 "confidence": round(c, 2), "bbox": xyxy}
            )
        elif name in (_HARDHAT, _NO_HARDHAT, _VEST, _NO_VEST):
            signals.append({"name": name, "conf": c, "center": _center(xyxy)})

    for obj in objects:
        if obj["type"] != "person":
            continue
        helmet = _best_signal(signals, obj["bbox"], _HARDHAT, _NO_HARDHAT)
        vest = _best_signal(signals, obj["bbox"], _VEST, _NO_VEST)
        obj["ppe"] = {"helmet": helmet, "safety_vest": vest}
    return objects


def _best_signal(signals, person_box, positive: str, negative: str) -> bool:
    pos = max((s["conf"] for s in signals
               if s["name"] == positive and _inside(s["center"], person_box)),
              default=0.0)
    neg = max((s["conf"] for s in signals
               if s["name"] == negative and _inside(s["center"], person_box)),
              default=0.0)
    return pos > neg


def draw_detections(rgb_image, detections: List[Dict]):
    import cv2

    out = np.array(rgb_image).copy()
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.4
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = str(det["id"])
        color = _VEHICLE_COLOR
        if det["type"] == "person":
            ppe = det.get("ppe", {})
            missing = not ppe.get("helmet", False) or not ppe.get("safety_vest", False)
            color = _ALERT_COLOR if missing else _PERSON_COLOR
            label += f" H{'y' if ppe.get('helmet') else 'n'}V{'y' if ppe.get('safety_vest') else 'n'}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        y_top = max(0, y1 - th - 3)
        cv2.rectangle(out, (x1, y_top), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, y1 - 2), font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out

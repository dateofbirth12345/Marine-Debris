from __future__ import annotations

import os
from typing import List, Optional
from PIL import Image
from ultralytics import YOLO

try:
    from .utils import load_config  # when imported as part of package `src`
    from .geo import extract_gps_from_image
except Exception:
    from utils import load_config  # fallback for direct script execution
    from geo import extract_gps_from_image


def load_best_model(cfg: dict) -> YOLO:
    best_model_path_file = os.path.join(cfg["paths"]["models_dir"], "best_model.path")
    if os.path.isfile(best_model_path_file):
        with open(best_model_path_file, "r", encoding="utf-8") as f:
            best_model_path = f.read().strip()
    else:
        # Fallback to configured model (for zero-shot usage)
        best_model_path = cfg["training"]["model"]
    return YOLO(best_model_path)


def predict_on_images(image_paths: List[str], conf: Optional[float] = None):
    cfg = load_config("configs/config.yaml")
    model = load_best_model(cfg)
    conf_thr = conf if conf is not None else float(cfg["inference"]["conf"])  # type: ignore
    iou_thr = float(cfg["inference"]["iou"])  # type: ignore
    results = model.predict(
        image_paths,
        imgsz=int(cfg["training"]["imgsz"]),
        conf=conf_thr,
        iou=iou_thr,
        verbose=False,
    )

    outputs = []
    for img_path, res in zip(image_paths, results):
        # Extract GPS if present
        gps = extract_gps_from_image(img_path)
        boxes = []
        names = res.names if hasattr(res, "names") else {}
        for b in res.boxes:  # type: ignore[attr-defined]
            cls_id = int(b.cls[0].item())
            boxes.append({
                "xyxy": b.xyxy[0].tolist(),  # [x1,y1,x2,y2]
                "conf": float(b.conf[0].item()),
                "cls": cls_id,
                "label": names.get(cls_id, str(cls_id)),
            })
        outputs.append({
            "image_path": img_path,
            "gps": gps,  # (lat, lon) or None
            "boxes": boxes,
        })
    return outputs


if __name__ == "__main__":
    # Example
    test_images = [
        # Add paths here for manual testing
    ]
    preds = predict_on_images(test_images)
    print(preds)



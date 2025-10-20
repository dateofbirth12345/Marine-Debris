from __future__ import annotations

import os
from ultralytics import YOLO

from utils import load_config


def main():
    cfg = load_config("configs/config.yaml")
    data_yaml = os.path.join("configs", "dataset.yolov8.yaml")

    # Load best model
    best_model_path_file = os.path.join(cfg["paths"]["models_dir"], "best_model.path")
    if os.path.isfile(best_model_path_file):
        with open(best_model_path_file, "r", encoding="utf-8") as f:
            best_model_path = f.read().strip()
    else:
        raise SystemExit("best_model.path not found. Train the model first.")

    model = YOLO(best_model_path)
    metrics = model.val(data=data_yaml)

    # Print key metrics
    # For detection: metrics.box.map, metrics.box.map50, metrics.box.map75, metrics.box.recall, etc.
    try:
        mp = metrics.box.map  # type: ignore[attr-defined]
        mp50 = metrics.box.map50  # type: ignore[attr-defined]
        recall = metrics.box.recall  # type: ignore[attr-defined]
        print(f"mAP: {mp:.4f}, mAP50: {mp50:.4f}, Recall: {recall:.4f}")
    except Exception:
        print(metrics)


if __name__ == "__main__":
    main()





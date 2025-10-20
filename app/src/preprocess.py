from __future__ import annotations

import os
from typing import List

from utils import load_config, find_yolo_pairs, split_dataset, materialize_splits, write_yolo_dataset_yaml


def main():
    cfg = load_config("configs/config.yaml")
    dataset_root = cfg["dataset"]["root"]
    classes: List[str] = cfg["dataset"].get("classes", ["debris"])  # default single class
    val_ratio = float(cfg["preprocess"]["val_ratio"])  # type: ignore
    test_ratio = float(cfg["preprocess"]["test_ratio"])  # type: ignore
    seed = int(cfg["preprocess"]["seed"])  # type: ignore
    splits_dir = cfg["paths"]["splits_dir"]

    os.makedirs(splits_dir, exist_ok=True)

    pairs = find_yolo_pairs(dataset_root)
    if len(pairs) == 0:
        raise SystemExit(
            "No YOLO-formatted dataset found. Expected 'images/' and 'labels/' sibling folders with .txt labels."
        )

    splits = split_dataset(pairs, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    materialize_splits(splits, splits_dir)

    dataset_yaml_path = os.path.join("configs", "dataset.yolov8.yaml")
    write_yolo_dataset_yaml(dataset_yaml_path, splits_dir, classes)
    print(f"Prepared splits at {splits_dir} and dataset yaml at {dataset_yaml_path}")


if __name__ == "__main__":
    main()





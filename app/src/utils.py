from __future__ import annotations

import os
import shutil
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import yaml


SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SplitPaths:
    train_images: str
    train_labels: str
    val_images: str
    val_labels: str
    test_images: str
    test_labels: str


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def list_images(root: str) -> List[str]:
    imgs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SUPPORTED_IMG_EXTS:
                imgs.append(os.path.join(dirpath, fn))
    return imgs


def find_yolo_pairs(dataset_root: str) -> List[Tuple[str, str]]:
    """Find YOLO image/label pairs under a dataset root.
    Looks for sibling folders named images/ and labels/.
    """
    candidates: List[Tuple[str, str]] = []
    for dirpath, dirnames, _ in os.walk(dataset_root):
        if os.path.basename(dirpath).lower() == "images":
            labels_dir = os.path.join(os.path.dirname(dirpath), "labels")
            if os.path.isdir(labels_dir):
                # pair files by stem
                for fn in os.listdir(dirpath):
                    stem, ext = os.path.splitext(fn)
                    if ext.lower() in SUPPORTED_IMG_EXTS:
                        img_path = os.path.join(dirpath, fn)
                        lbl_path = os.path.join(labels_dir, stem + ".txt")
                        if os.path.isfile(lbl_path):
                            candidates.append((img_path, lbl_path))
    return candidates


def split_dataset(pairs: List[Tuple[str, str]], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Tuple[str, str]]]:
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_set = pairs[:n_test]
    val_set = pairs[n_test:n_test + n_val]
    train_set = pairs[n_test + n_val:]
    return {"train": train_set, "val": val_set, "test": test_set}


def materialize_splits(splits: Dict[str, List[Tuple[str, str]]], target_dir: str) -> SplitPaths:
    train_images = os.path.join(target_dir, "train", "images")
    train_labels = os.path.join(target_dir, "train", "labels")
    val_images = os.path.join(target_dir, "val", "images")
    val_labels = os.path.join(target_dir, "val", "labels")
    test_images = os.path.join(target_dir, "test", "images")
    test_labels = os.path.join(target_dir, "test", "labels")
    ensure_dirs([train_images, train_labels, val_images, val_labels, test_images, test_labels])

    for split_name, items in splits.items():
        for img_path, lbl_path in items:
            dst_img_dir = locals()[f"{split_name}_images"]
            dst_lbl_dir = locals()[f"{split_name}_labels"]
            shutil.copy2(img_path, os.path.join(dst_img_dir, os.path.basename(img_path)))
            shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, os.path.basename(lbl_path)))

    return SplitPaths(
        train_images=train_images,
        train_labels=train_labels,
        val_images=val_images,
        val_labels=val_labels,
        test_images=test_images,
        test_labels=test_labels,
    )


def write_yolo_dataset_yaml(dataset_yaml_path: str, splits_dir: str, class_names: List[str]) -> None:
    data = {
        "path": os.path.abspath(splits_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    os.makedirs(os.path.dirname(dataset_yaml_path), exist_ok=True)
    with open(dataset_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)





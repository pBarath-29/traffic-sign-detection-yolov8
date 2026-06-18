from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import yaml
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "Traffic_Signs_Detection" / "car"
VIDEO_PATH = BASE / "Traffic_Signs_Detection" / "video.mp4"

# 15 perceptually distinct colors for the 15 classes
CLASS_COLORS = [
    "#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12",
    "#1abc9c", "#e67e22", "#d35400", "#c0392b", "#16a085",
    "#8e44ad", "#2980b9", "#27ae60", "#f1c40f", "#7f8c8d",
]


def get_class_names() -> List[str]:
    with open(DATASET / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["names"]


def _split_dir(split: str) -> Path:
    return DATASET / split


def load_split(split: str) -> pd.DataFrame:
    """Parse all YOLO label files for a given split into a DataFrame."""
    class_names = get_class_names()
    labels_dir = _split_dir(split) / "labels"
    images_dir = _split_dir(split) / "images"

    records = []
    for txt in labels_dir.glob("*.txt"):
        stem = txt.stem
        img_candidates = list(images_dir.glob(f"{stem}.*"))
        img_path = str(img_candidates[0]) if img_candidates else ""
        content = txt.read_text().strip()
        if not content:
            continue
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            records.append({
                "split": split,
                "image_stem": stem,
                "image_path": img_path,
                "class_id": cid,
                "class_name": class_names[cid] if cid < len(class_names) else str(cid),
                "cx": cx, "cy": cy, "w": w, "h": h,
                "area": w * h,
            })

    if not records:
        return pd.DataFrame(columns=["split", "image_stem", "image_path",
                                     "class_id", "class_name", "cx", "cy", "w", "h", "area"])
    return pd.DataFrame(records)


def load_all_splits() -> pd.DataFrame:
    parts = []
    for split in ("train", "valid", "test"):
        try:
            parts.append(load_split(split))
        except FileNotFoundError:
            pass
    return pd.concat(parts, ignore_index=True)


def get_image_path(stem: str, split: str) -> Optional[Path]:
    images_dir = _split_dir(split) / "images"
    candidates = list(images_dir.glob(f"{stem}.*"))
    return candidates[0] if candidates else None


def color_for_class(class_id: int) -> str:
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def load_image_with_boxes(stem: str, split: str, df: pd.DataFrame,
                          img_size: int = 416) -> np.ndarray:
    """Load an image and draw YOLO bounding boxes on it. Returns RGB array."""
    path = get_image_path(stem, split)
    if path is None:
        blank = np.full((img_size, img_size, 3), 40, dtype=np.uint8)
        return blank

    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    draw = ImageDraw.Draw(img)
    rows = df[(df["image_stem"] == stem) & (df["split"] == split)]
    W, H = img.size

    for _, row in rows.iterrows():
        cx, cy, w, h = row["cx"], row["cy"], row["w"], row["h"]
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)
        color = color_for_class(int(row["class_id"]))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = row["class_name"]
        draw.text((x1 + 2, max(y1 - 12, 0)), label, fill=color)

    return np.array(img)


def build_heatmap_array(df: pd.DataFrame, split: Optional[str] = None,
                        class_filter: Optional[List[int]] = None,
                        grid_size: int = 64) -> np.ndarray:
    """Accumulate bbox coverage into a grid density map."""
    subset = df.copy()
    if split:
        subset = subset[subset["split"] == split]
    if class_filter:
        subset = subset[subset["class_id"].isin(class_filter)]

    density = np.zeros((grid_size, grid_size), dtype=np.float32)
    for _, row in subset.iterrows():
        cx, cy, w, h = row["cx"], row["cy"], row["w"], row["h"]
        x1 = max(0, int((cx - w / 2) * grid_size))
        y1 = max(0, int((cy - h / 2) * grid_size))
        x2 = min(grid_size, int((cx + w / 2) * grid_size))
        y2 = min(grid_size, int((cy + h / 2) * grid_size))
        density[y1:y2, x1:x2] += 1.0

    if density.max() > 0:
        density /= density.max()
    return density


def get_annotation_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-class annotation counts for each split."""
    class_names = get_class_names()
    rows = []
    for split in df["split"].unique():
        counts = df[df["split"] == split]["class_name"].value_counts()
        for cid, name in enumerate(class_names):
            rows.append({
                "split": split,
                "class_id": cid,
                "class_name": name,
                "count": int(counts.get(name, 0)),
            })
    return pd.DataFrame(rows)

"""
Offline YOLOv8 training script.

Run from the project root:
    cd "C:\\Users\\P Barath\\Downloads\\BENG_RMI_01"
    python scripts/train.py

Outputs:
    outputs/yolov8n_traffic/weights/best.pt
    outputs/yolov8n_traffic/weights/last.pt
    outputs/yolov8n_traffic/results.csv
    outputs/metrics.json
    outputs/confidence_detections.csv
"""

import json
import sys
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_YAML = ROOT / "Traffic_Signs_Detection" / "car" / "data.yaml"
OUTPUT_DIR = ROOT / "outputs"
RUN_NAME = "yolov8n_traffic"


def patch_data_yaml():
    """Replace relative paths in data.yaml with absolute paths so training works
    regardless of the working directory."""
    import yaml
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)

    car_dir = DATA_YAML.parent
    patched = ROOT / "outputs" / "data_patched.yaml"
    patched.parent.mkdir(parents=True, exist_ok=True)

    cfg["train"] = str(car_dir / "train" / "images")
    cfg["val"]   = str(car_dir / "valid" / "images")
    cfg["test"]  = str(car_dir / "test"  / "images")

    with open(patched, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return str(patched)


def export_metrics(model, patched_yaml: str):
    """Run test-set evaluation and export per-class metrics + confusion matrix to JSON."""
    print("\n[Evaluation] Running on test split…")
    metrics = model.val(data=patched_yaml, split="test", verbose=False)
    box = metrics.box

    class_names = model.names
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in sorted(class_names.keys())]

    precision = box.p.tolist() if hasattr(box, "p") else []
    recall    = box.r.tolist() if hasattr(box, "r") else []
    ap50      = box.ap50.tolist() if hasattr(box, "ap50") else []
    ap        = box.ap.tolist() if hasattr(box, "ap") else []
    map50     = float(box.map50) if hasattr(box, "map50") else 0.0
    map_val   = float(box.map)   if hasattr(box, "map")   else 0.0

    # Build PR curves (ultralytics stores them in metrics.box.curves_results or similar)
    pr_curves = {}
    if hasattr(box, "curves_results"):
        # ultralytics >= 8.1 exposes per-class curves
        for i, name in enumerate(class_names):
            try:
                pr_curves[name] = {
                    "recall":    box.curves_results[0][i].tolist(),
                    "precision": box.curves_results[1][i].tolist(),
                }
            except Exception:
                pass

    # Confusion matrix
    cm_data = None
    if hasattr(metrics, "confusion_matrix") and metrics.confusion_matrix is not None:
        cm = metrics.confusion_matrix.matrix
        if cm is not None:
            cm_data = cm.tolist()

    output = {
        "class_names": class_names,
        "precision":   precision,
        "recall":      recall,
        "ap50":        ap50,
        "ap":          ap,
        "map50":       map50,
        "map":         map_val,
        "pr_curves":   pr_curves,
        "confusion_matrix": cm_data,
    }

    out_path = OUTPUT_DIR / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[Evaluation] Metrics saved → {out_path}")
    print(f"  mAP@50      = {map50:.4f}")
    print(f"  mAP@50-95   = {map_val:.4f}")

    # Export confidence detections CSV for violin plot
    _export_confidence_csv(model, patched_yaml, class_names)


def _export_confidence_csv(model, patched_yaml: str, class_names: list):
    """Run inference on test images and collect confidence scores per class."""
    from ultralytics.data import build_dataloader
    import cv2

    test_img_dir = ROOT / "Traffic_Signs_Detection" / "car" / "test" / "images"
    img_paths = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    img_paths = img_paths[:400]  # cap at 400 for speed

    records = []
    for img_path in img_paths:
        results = model(str(img_path), conf=0.1, verbose=False)
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                name = class_names[cid] if cid < len(class_names) else str(cid)
                records.append({"class_id": cid, "class_name": name, "confidence": conf})

    if records:
        out_path = OUTPUT_DIR / "confidence_detections.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "confidence"])
            writer.writeheader()
            writer.writerows(records)
        print(f"[Evaluation] Confidence CSV saved → {out_path} ({len(records)} detections)")


def main():
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML.exists():
        print(f"ERROR: data.yaml not found at {DATA_YAML}")
        print("Ensure the Traffic_Signs_Detection/car/ directory is in the project root.")
        sys.exit(1)

    patched_yaml = patch_data_yaml()
    print(f"[Setup] Patched data.yaml → {patched_yaml}")

    print("\n[Training] Loading YOLOv8n base weights…")
    model = YOLO("yolov8n.pt")

    print("[Training] Starting fine-tuning…")
    model.train(
        data=patched_yaml,
        epochs=50,
        imgsz=416,
        batch=16,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        patience=10,
        augment=True,
        cls=1.5,           # upweight classification loss for imbalanced dataset
        degrees=10.0,      # rotation augmentation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        exist_ok=True,
        verbose=True,
    )

    best_weights = OUTPUT_DIR / RUN_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        print("ERROR: Training did not produce best.pt. Check logs above.")
        sys.exit(1)

    print(f"\n[Training] Complete. Best weights → {best_weights}")

    # Reload best weights for evaluation
    model = YOLO(str(best_weights))
    export_metrics(model, patched_yaml)

    print("\n[Done] All outputs saved to outputs/")
    print("  Launch the dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()

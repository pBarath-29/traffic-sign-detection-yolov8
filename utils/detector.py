"""YOLOv8 inference wrapper. Kept separate so Streamlit cache decorators work cleanly."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2

from utils.data_loader import CLASS_COLORS, get_class_names, BASE

WEIGHTS_DIR = BASE / "outputs" / "yolov8n_traffic" / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"  # pre-trained COCO weights


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def load_model(weights_path: Optional[str] = None):
    """Load a YOLO model. Returns (model, is_custom) tuple."""
    from ultralytics import YOLO
    if weights_path is None:
        if DEFAULT_WEIGHTS.exists():
            weights_path = str(DEFAULT_WEIGHTS)
        else:
            weights_path = FALLBACK_WEIGHTS
    model = YOLO(weights_path)
    is_custom = Path(weights_path).name == "best.pt" and DEFAULT_WEIGHTS.exists()
    return model, is_custom


def run_inference(model, image: np.ndarray, conf: float = 0.25) -> List[Dict]:
    """Run YOLOv8 inference on an RGB numpy array. Returns list of detection dicts."""
    class_names = get_class_names()
    results = model(image, conf=conf, verbose=False)
    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cid = int(box.cls[0].item())
            detections.append({
                "class_id": cid,
                "class_name": class_names[cid] if cid < len(class_names) else str(cid),
                "confidence": float(box.conf[0].item()),
                "x1": float(box.xyxy[0][0].item()),
                "y1": float(box.xyxy[0][1].item()),
                "x2": float(box.xyxy[0][2].item()),
                "y2": float(box.xyxy[0][3].item()),
            })
    return detections


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes on an RGB image. Returns RGB array."""
    out = image.copy()
    for det in detections:
        cid = det["class_id"]
        color_hex = CLASS_COLORS[cid % len(CLASS_COLORS)]
        color_bgr = _hex_to_bgr(color_hex)

        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])

        # Convert RGB→BGR for OpenCV, draw, convert back
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color_bgr, 2)
        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return out


_SIGN_HEIGHTS_M: Dict[str, float] = {
    "Stop": 0.60,
    "Red Light": 0.80,
    "Green Light": 0.80,
}
_DEFAULT_SIGN_HEIGHT_M = 0.60
_ASSUMED_FOCAL_PX = 416  # 90° HFOV dashcam at 416-px resolution


def estimate_distance(det: Dict, focal_px: int = _ASSUMED_FOCAL_PX) -> Optional[float]:
    """Estimate sign distance (metres) via pinhole camera model.
    d = (real_height_m * focal_px) / bbox_height_px"""
    bbox_h = det["y2"] - det["y1"]
    if bbox_h <= 0:
        return None
    real_h = _SIGN_HEIGHTS_M.get(det["class_name"], _DEFAULT_SIGN_HEIGHT_M)
    return (real_h * focal_px) / bbox_h


def get_gradcam(model, img_rgb_np: np.ndarray) -> Optional[np.ndarray]:
    """EigenCAM heatmap overlay targeting the YOLOv8 backbone SPPF layer (index 9).
    Returns an RGB numpy array, or None if pytorch-grad-cam is not installed."""
    try:
        from pytorch_grad_cam import EigenCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    from PIL import Image as _PILImg

    # pytorch-grad-cam expects the model to return a plain tensor, but YOLOv8's
    # DetectionModel returns a tuple. Wrap it so grad-cam gets a flat tensor.
    class _CAMWrapper(nn.Module):
        def __init__(self, det_model: nn.Module) -> None:
            super().__init__()
            self._m = det_model

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out = self._m(x)
            # unwrap tuple/list until we have a tensor
            while isinstance(out, (tuple, list)):
                out = out[0]
            # flatten [B, anchors, classes+4] -> [B, N] so argmax works
            if out.dim() == 3:
                out = out.reshape(out.shape[0], -1)
            return out

    h, w = img_rgb_np.shape[:2]
    img_416 = np.array(_PILImg.fromarray(img_rgb_np).resize((416, 416))).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_416).permute(2, 0, 1).unsqueeze(0)

    wrapper = _CAMWrapper(model.model)
    target_layer = [model.model.model[9]]  # SPPF backbone bottleneck
    cam = EigenCAM(wrapper, target_layer)
    grayscale_cam = cam(input_tensor=img_tensor)[0]
    cam_resized = (
        np.array(_PILImg.fromarray((grayscale_cam * 255).astype(np.uint8)).resize((w, h))) / 255.0
    )
    return show_cam_on_image(img_rgb_np.astype(np.float32) / 255.0, cam_resized, use_rgb=True)


def run_video_inference(model, video_path: str, output_path: str,
                        conf: float = 0.25) -> dict:
    """Frame-by-frame inference on a video. Writes annotated output video.
    Returns stats dict with detections_per_frame and class_counts."""
    class_names = get_class_names()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # avc1 = H.264, browser-compatible; fall back to mp4v if unavailable
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    detections_per_frame = []
    class_counts: dict[str, int] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = run_inference(model, frame_rgb, conf=conf)
        detections_per_frame.append(len(dets))
        for d in dets:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

        annotated = draw_detections(frame_rgb, dets)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)

    cap.release()
    out.release()

    return {
        "total_frames": total_frames,
        "fps": fps,
        "detections_per_frame": detections_per_frame,
        "class_counts": class_counts,
    }

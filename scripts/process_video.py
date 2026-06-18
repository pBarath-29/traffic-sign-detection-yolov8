"""
Offline video inference script.

Run from the project root AFTER training:
    cd "C:\\Users\\P Barath\\Downloads\\BENG_RMI_01"
    python scripts/process_video.py

Outputs:
    outputs/processed_video.mp4    — annotated dashcam video
    outputs/video_stats.json       — per-frame detection statistics
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.data_loader import VIDEO_PATH
from utils.detector import load_model, run_video_inference

VIDEO_IN  = VIDEO_PATH
VIDEO_OUT = ROOT / "outputs" / "processed_video.mp4"
STATS_OUT = ROOT / "outputs" / "video_stats.json"
CONF      = 0.25


def main():
    if not VIDEO_IN.exists():
        print(f"ERROR: Video not found at {VIDEO_IN}")
        sys.exit(1)

    print(f"[Video] Input  → {VIDEO_IN}")
    print(f"[Video] Output → {VIDEO_OUT}")

    VIDEO_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("[Model] Loading weights…")
    model, is_custom = load_model()
    if is_custom:
        print("[Model] Using custom fine-tuned weights (best.pt)")
    else:
        print("[Model] WARNING: Custom weights not found. Using COCO pre-trained fallback.")
        print("         Run scripts/train.py first for full 15-class detection.")

    print(f"[Inference] Processing video (conf threshold = {CONF})…")
    stats = run_video_inference(
        model,
        video_path=str(VIDEO_IN),
        output_path=str(VIDEO_OUT),
        conf=CONF,
    )

    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    total_det = sum(stats["detections_per_frame"])
    print(f"\n[Done] Processed {stats['total_frames']} frames at {stats['fps']:.1f} FPS")
    print(f"       Total detections: {total_det}")
    print(f"       Class breakdown:")
    for cls, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        print(f"         {cls:25s}: {count}")
    print(f"\n[Output] Video  → {VIDEO_OUT}")
    print(f"[Output] Stats  → {STATS_OUT}")
    print("\nLaunch the dashboard to view:  streamlit run app.py")


if __name__ == "__main__":
    main()

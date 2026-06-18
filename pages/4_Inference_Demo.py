import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
from pathlib import Path
from PIL import Image

from utils.data_loader import BASE, VIDEO_PATH
from utils.detector import load_model, run_inference, draw_detections, get_gradcam, estimate_distance
from utils.visualizer import plot_detections_per_frame, plot_detection_donut

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
pio.templates.default = "plotly_white"

PROCESSED_VIDEO = BASE / "outputs" / "processed_video_h264.mp4"
ORIGINAL_VIDEO  = BASE / "outputs" / "original_video_h264.mp4"
VIDEO_STATS     = BASE / "outputs" / "video_stats.json"
WEIGHTS_PATH    = BASE / "outputs" / "yolov8n_traffic" / "weights" / "best.pt"

st.title("Inference Demo")
st.markdown("Before and after comparison of the dashcam video, plus live detection on uploaded images.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Before / After Video ──────────────────────────────────────────────────────

st.header("Before & After — Dashcam Video")

col_before, col_after = st.columns(2)

with col_before:
    st.markdown('<p class="video-label">Original</p>', unsafe_allow_html=True)
    if ORIGINAL_VIDEO.exists():
        st.video(ORIGINAL_VIDEO.read_bytes())
    elif VIDEO_PATH.exists():
        st.video(VIDEO_PATH.read_bytes())
    else:
        st.info("Original video not found.")

with col_after:
    st.markdown('<p class="video-label">With Detections</p>', unsafe_allow_html=True)
    if PROCESSED_VIDEO.exists():
        st.video(PROCESSED_VIDEO.read_bytes())
    else:
        st.markdown("""
<div class="warning-box">
Processed video not found. Run:<br/>
<code>python scripts/process_video.py</code>
</div>
""", unsafe_allow_html=True)

# ── Video Stats ───────────────────────────────────────────────────────────────

if VIDEO_STATS.exists():
    with open(VIDEO_STATS) as f:
        stats = json.load(f)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Frames", stats.get("total_frames", "—"))
    c2.metric("FPS", f"{stats.get('fps', 0):.1f}")
    c3.metric("Total Detections", sum(stats.get("detections_per_frame", [])))
    cc = stats.get("class_counts", {})
    top = max(cc, key=cc.get) if cc else "—"
    c4.metric("Most Detected", top)

    st.plotly_chart(plot_detections_per_frame(stats), use_container_width=True)

    if cc:
        st.subheader("Detection Breakdown")
        total_det = sum(cc.values())
        bd_df = pd.DataFrame([
            {"Class": k, "Count": v, "%": f"{100*v/total_det:.1f}%"}
            for k, v in sorted(cc.items(), key=lambda x: -x[1])
        ])
        st.dataframe(bd_df, use_container_width=True, hide_index=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Upload Image ──────────────────────────────────────────────────────────────

st.header("Upload an Image")
st.markdown("Upload any traffic scene image to see live detections.")

if not WEIGHTS_PATH.exists():
    st.markdown("""
<div class="warning-box">
Custom model weights not found — using the base YOLOv8n (COCO) as fallback.
It recognises only <em>traffic light</em> and <em>stop sign</em>.
Run <code>python scripts/train.py</code> to enable all 15 classes.
</div>
""", unsafe_allow_html=True)

col_up, col_ctrl = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader("Upload image (JPG / PNG)", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
with col_ctrl:
    conf_threshold = st.slider("Confidence", 0.05, 0.95, 0.25, 0.05)

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")
    img_np  = np.array(img_pil)

    with st.spinner("Running inference and computing neural attention…"):
        model, _ = load_model()
        detections = run_inference(model, img_np, conf=conf_threshold)
        annotated  = draw_detections(img_np, detections)
        cam_img    = get_gradcam(model, img_np)

    if cam_img is not None:
        col_orig, col_det, col_cam = st.columns(3)
    else:
        col_orig, col_det = st.columns(2)
        col_cam = None

    with col_orig:
        st.markdown('<p class="video-label">Original</p>', unsafe_allow_html=True)
        st.image(img_np, use_container_width=True)
    with col_det:
        st.markdown('<p class="video-label">Detections</p>', unsafe_allow_html=True)
        st.image(annotated, use_container_width=True)
    if col_cam is not None:
        with col_cam:
            st.markdown('<p class="video-label">Neural Attention</p>', unsafe_allow_html=True)
            st.image(cam_img, use_container_width=True)

    if cam_img is not None:
        st.markdown("""
<div class="insight-box">
<strong>Neural Attention (EigenCAM)</strong> visualises which spatial regions the YOLOv8 backbone
activates most strongly. Hot areas (red/yellow) drove the detection predictions.
Target: SPPF layer — the backbone feature bottleneck at depth 9.
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if detections:
        st.success(f"{len(detections)} object(s) detected above confidence {conf_threshold:.2f}")

        col_tbl, col_donut = st.columns([2, 1])
        with col_tbl:
            _rows = []
            for d in sorted(detections, key=lambda x: -x["confidence"]):
                _dist = estimate_distance(d)
                _rows.append({
                    "Class": d["class_name"],
                    "Confidence": f"{d['confidence']:.3f}",
                    "Width (px)": int(d["x2"] - d["x1"]),
                    "Height (px)": int(d["y2"] - d["y1"]),
                    "Dist (m)": f"{_dist:.1f}" if _dist is not None else "—",
                })
            det_df = pd.DataFrame(_rows)
            st.dataframe(det_df, use_container_width=True, hide_index=True)
        st.markdown("""
<div class="insight-box">
<strong>Distance estimation</strong> uses the pinhole camera model:
<code>d = (real_height_m × focal_px) / bbox_height_px</code>.
Focal length assumed 416 px (90° HFOV dashcam at 416-px resolution).
Sign heights: traffic lights = 0.8 m, all others = 0.6 m (EU standard).
</div>
""", unsafe_allow_html=True)

        with col_donut:
            st.plotly_chart(plot_detection_donut(detections), use_container_width=True)

        # Driving decisions
        st.subheader("Decision Output")
        for det in sorted(detections, key=lambda x: -x["confidence"]):
            name, conf = det["class_name"], det["confidence"]
            if name == "Red Light":
                st.error(f"Red Light ({conf:.2f}) → Stop")
            elif name == "Green Light":
                st.success(f"Green Light ({conf:.2f}) → Proceed")
            elif name == "Stop":
                st.error(f"Stop Sign ({conf:.2f}) → Full stop")
            elif "Speed Limit" in name:
                limit = name.replace("Speed Limit ", "")
                st.info(f"{name} ({conf:.2f}) → Cap speed at {limit} km/h")
    else:
        st.warning(f"No detections above {conf_threshold:.2f}. Try lowering the confidence slider.")

else:
    st.markdown("""
<div style="background:#f9fafb; border:1px dashed #d1d5db; border-radius:8px;
padding:40px; text-align:center; color:#9ca3af;">
Upload an image to see detections
</div>
""", unsafe_allow_html=True)

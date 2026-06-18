from __future__ import annotations
import streamlit as st
import plotly.io as pio
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Traffic Sign Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

pio.templates.default = "plotly_white"

# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-container">
  <h1 style="margin:0; font-size:2rem;">Traffic Sign Detection</h1>
  <p class="hero-subtitle">
    End-to-end YOLOv8 pipeline trained on 4,969 real dashcam images across 15 traffic sign classes.
  </p>
  <br/>
  <span class="hero-badge">YOLOv8</span>
  <span class="hero-badge">Computer Vision</span>
  <span class="hero-badge">Object Detection</span>
  <span class="hero-badge">4,969 Images</span>
  <span class="hero-badge">15 Classes</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Images", "4,969")
c2.metric("Classes", "15")
c3.metric("mAP@50", "0.932")
c4.metric("Image Resolution", "416 × 416")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Overview ──────────────────────────────────────────────────────────────────

col_l, col_r = st.columns(2)

with col_l:
    st.markdown("""
### What this project does

Trains and evaluates a YOLOv8 object detection model on a real dashcam traffic sign dataset,
then runs inference on a dashcam video to produce annotated output.

**Pipeline:**

| Stage | Description |
|-------|-------------|
| Data Analysis | Explore 4,969 annotated images — class distribution, spatial patterns, sign sizes |
| Model Training | Fine-tune YOLOv8n with augmentation and imbalance-aware loss |
| Evaluation | Per-class precision, recall, mAP, confusion matrix, PR curves |
| Inference | Run detection on dashcam video — before and after comparison |
""")

with col_r:
    st.markdown("""
### Dataset — 15 Traffic Sign Classes

**Traffic Lights**
- Green Light · Red Light

**Speed Limits**
- 10 · 20 · 30 · 40 · 50 · 60 · 70 · 80 · 90 · 100 · 110 · 120 km/h

**Regulatory**
- Stop

**Dataset split:** 3,530 train / 801 val / 638 test

Source: Roboflow Self-Driving Cars Traffic Signs · CC BY 4.0
""")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

_VAL_PREVIEW = Path(__file__).parent / "outputs" / "yolov8n_traffic" / "val_batch0_pred.jpg"
if _VAL_PREVIEW.exists():
    st.subheader("Sample Predictions")
    st.markdown(
        "YOLOv8 predictions on held-out validation images — bounding boxes and class labels "
        "predicted by the trained model, never seen during training."
    )
    st.image(Image.open(_VAL_PREVIEW), use_container_width=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.markdown("""
### Navigate

Use the **sidebar** to explore each section:

- **Dataset Explorer** — class distribution, spatial heatmaps, annotated image grid
- **Model & Training** — architecture breakdown, training config, loss curves
- **Performance** — mAP radar chart, confusion matrix, precision-recall curves
- **Inference Demo** — before & after dashcam video, live image upload
- **Insights** — key findings and engineering analysis
""")

st.caption("Dataset: Roboflow Self-Driving Cars Traffic Signs · CC BY 4.0")

import random
import streamlit as st
import plotly.io as pio
from pathlib import Path

from utils.data_loader import load_all_splits, get_annotation_counts, build_heatmap_array, get_class_names, CLASS_COLORS
from utils.visualizer import (
    plot_class_distribution, plot_spatial_heatmap, plot_bbox_size_scatter,
    plot_donut_imbalance, plot_mosaic,
)

# Inject CSS
css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
pio.templates.default = "plotly_white"

st.title("Dataset Explorer")
st.markdown(
    "Deep-dive into the **4,969-image Traffic Signs dataset**: distribution, "
    "spatial patterns, sign sizes, and the critical class imbalance challenge."
)

# ── Load data ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Parsing 4,969 YOLO label files…")
def load_data():
    df = load_all_splits()
    counts = get_annotation_counts(df)
    return df, counts

df, counts = load_data()
total_annotations = len(df)
unique_images = df["image_stem"].nunique()

# ── Quick Stats ──────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Images", "4,969")
c2.metric("Annotated Images", f"{unique_images:,}")
c3.metric("Total Annotations", f"{total_annotations:,}")
c4.metric("Classes", "15")
c5.metric("Imbalance Ratio", "30.8×", "Speed Limit 10 vs Red Light")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Class Distribution ───────────────────────────────────────────────────────

st.header("Class Distribution")
st.markdown(
    "Annotation counts across all three dataset splits. "
    "Hover over bars to see exact counts."
)
st.plotly_chart(plot_class_distribution(counts), use_container_width=True)

st.markdown("""
<div class="warning-box">
⚠️ <strong>Class Imbalance Detected:</strong> Speed Limit 10 has only <strong>19 training annotations</strong>
compared to 585 for Red Light — a <strong>30.8× imbalance</strong>. This is a real-world engineering challenge
that motivates weighted loss functions, copy-paste augmentation, and few-shot learning techniques.
Speed Limit 10 also has <strong>zero validation examples</strong>, making its mAP effectively 0.
</div>
""", unsafe_allow_html=True)

# Imbalance table
st.subheader("Per-Class Annotation Summary")
train_counts = counts[counts["split"] == "train"].set_index("class_name")["count"]
valid_counts = counts[counts["split"] == "valid"].set_index("class_name")["count"]
test_counts = counts[counts["split"] == "test"].set_index("class_name")["count"]
class_names = get_class_names()
summary_rows = []
for name in class_names:
    tr = int(train_counts.get(name, 0))
    va = int(valid_counts.get(name, 0))
    te = int(test_counts.get(name, 0))
    total = tr + va + te
    max_count = int(train_counts.max())
    ratio = f"1 : {max_count / max(tr, 1):.0f}" if tr < max_count else "—"
    summary_rows.append({"Class": name, "Train": tr, "Val": va, "Test": te, "Total": total, "Imbalance vs Max": ratio})

import pandas as pd
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df.style.background_gradient(subset=["Train"], cmap="YlOrRd"), use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Donut Chart ──────────────────────────────────────────────────────────────

st.header("Split Composition")
col_d1, col_d2, col_d3 = st.columns(3)
for col, split in zip([col_d1, col_d2, col_d3], ["train", "valid", "test"]):
    with col:
        st.plotly_chart(plot_donut_imbalance(counts, split=split), use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Spatial Heatmap ──────────────────────────────────────────────────────────

st.header("Bounding Box Spatial Density")
st.markdown(
    "Where do traffic signs actually appear within the camera frame? "
    "Brighter regions indicate higher annotation density."
)

col_ctrl, col_heat = st.columns([1, 3])
with col_ctrl:
    split_sel = st.selectbox("Split", ["train", "valid", "test", "all"], index=0)
    class_filter_names = st.multiselect(
        "Filter by class (all = no filter)",
        options=class_names,
        default=[],
    )
    grid_sz = st.slider("Heatmap resolution", 32, 128, 64, step=16)

split_arg = None if split_sel == "all" else split_sel
class_ids = [class_names.index(n) for n in class_filter_names] if class_filter_names else None

with st.spinner("Building heatmap…"):
    density = build_heatmap_array(df, split=split_arg, class_filter=class_ids, grid_size=grid_sz)

title_suf = f"{split_sel.capitalize()}"
if class_filter_names:
    title_suf += " — " + ", ".join(class_filter_names)

with col_heat:
    st.plotly_chart(plot_spatial_heatmap(density, title_suf), use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <strong>Insight:</strong> Signs cluster in the <strong>upper 40% and horizontal center</strong> of the frame —
consistent with forward-facing dashcam footage. This spatial prior can be leveraged to design
<em>region-of-interest cropping</em> before detection, reducing inference latency on edge hardware.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── BBox Size Scatter ────────────────────────────────────────────────────────

st.header("Bounding Box Size Distribution")
st.markdown(
    "Sign size (in pixels at 416×416 resolution) varies dramatically across classes. "
    "Small signs at distance motivate multi-scale feature pyramid detection."
)

split_scatter = st.selectbox("Split for scatter plot", ["train", "valid", "test"], index=0,
                             key="scatter_split")
scatter_df = df[df["split"] == split_scatter]
st.plotly_chart(plot_bbox_size_scatter(scatter_df), use_container_width=True)

st.markdown("""
<div class="insight-box">
💡 <strong>Insight:</strong> Median bounding box size is approximately <strong>30×30 px</strong> at 416-px resolution.
This places most signs in the "small object" category (< 32×32 px by COCO definition), explaining why
YOLOv8's <strong>Feature Pyramid Network neck</strong> is essential for detection — it fuses multi-scale
feature maps to handle objects at vastly different scales.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Image Mosaic ─────────────────────────────────────────────────────────────

st.header("Annotated Image Mosaic")
st.markdown("Random sample of training images with ground-truth bounding boxes overlaid (one color per class).")

col_m1, col_m2, col_m3 = st.columns([1, 1, 2])
with col_m1:
    mosaic_split = st.selectbox("Split", ["train", "valid", "test"], key="mosaic_split")
with col_m2:
    mosaic_cols = st.slider("Grid columns", 3, 6, 5, key="mosaic_cols")
with col_m3:
    resample = st.button("🔀  Resample images")

mosaic_n = mosaic_cols * 3
split_stems = df[df["split"] == mosaic_split]["image_stem"].unique().tolist()

if "mosaic_stems" not in st.session_state or resample or st.session_state.get("last_mosaic_split") != mosaic_split:
    st.session_state["mosaic_stems"] = random.sample(split_stems, min(mosaic_n, len(split_stems)))
    st.session_state["last_mosaic_split"] = mosaic_split

chosen_stems = st.session_state["mosaic_stems"][:mosaic_n]

with st.spinner("Rendering mosaic…"):
    mosaic = plot_mosaic(chosen_stems, mosaic_split, df, cols=mosaic_cols, img_size=200)

st.image(mosaic, caption=f"{len(chosen_stems)} images from {mosaic_split} split — click to enlarge",
         use_container_width=True)

# ── Legend ──────────────────────────────────────────────────────────────────

st.subheader("Class Color Legend")
cols = st.columns(5)
for i, name in enumerate(class_names):
    color = CLASS_COLORS[i % len(CLASS_COLORS)]
    cols[i % 5].markdown(
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{color};border-radius:2px;margin-right:6px;"></span>'
        f'<span style="font-size:0.85rem;color:#c0c0c0;">{name}</span>',
        unsafe_allow_html=True,
    )

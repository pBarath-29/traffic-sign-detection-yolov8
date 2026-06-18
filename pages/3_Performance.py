import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
from pathlib import Path

from utils.data_loader import BASE, get_class_names
from utils.visualizer import (
    plot_radar_map, plot_per_class_bar, plot_confusion_matrix,
    plot_pr_curves, plot_violin_confidence,
)

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
pio.templates.default = "plotly_white"

METRICS_JSON = BASE / "outputs" / "metrics.json"
CONF_CSV = BASE / "outputs" / "confidence_detections.csv"

st.title("Performance Dashboard")
st.markdown(
    "Per-class precision, recall, mAP, and confusion analysis on the held-out **test set** "
    "(638 images, never seen during training or validation)."
)


def load_metrics():
    if not METRICS_JSON.exists():
        return None
    with open(METRICS_JSON) as f:
        return json.load(f)


metrics = load_metrics()

if metrics is None:
    st.markdown("""
<div class="warning-box">
⚠️ <strong>No metrics found.</strong> Train the model first, then run evaluation:

```bash
python scripts/train.py   # generates outputs/metrics.json automatically
```

This page will display:
- Summary mAP, Precision, Recall metrics
- Per-class radar chart (the project's signature visual)
- Confusion matrix showing cross-class failures
- Precision-Recall curves for all 15 classes
- Confidence score distributions

**Expected highlights once trained:**
- Overall mAP@50 ≈ 0.80-0.85
- Green Light and Red Light: near-perfect recall (>0.90)
- Speed Limit 10: mAP ≈ 0 (only 19 training samples)
- Most confusions: Speed Limit 100 ↔ 110, Speed Limit 80 ↔ 90
</div>
""", unsafe_allow_html=True)

else:
    class_names = metrics["class_names"] if "class_names" in metrics else get_class_names()
    precision = metrics.get("precision", [])
    recall = metrics.get("recall", [])
    ap50 = metrics.get("ap50", [])
    ap = metrics.get("ap", [])
    map50 = metrics.get("map50", 0.0)
    map_val = metrics.get("map", 0.0)

    f1_scores = [2 * p * r / (p + r + 1e-8) for p, r in zip(precision, recall)]
    mean_precision = float(np.mean(precision)) if precision else 0.0
    mean_recall = float(np.mean(recall)) if recall else 0.0
    mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    # ── Summary metrics ──────────────────────────────────────────────────────

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("mAP@50", f"{map50:.3f}")
    m2.metric("mAP@50-95", f"{map_val:.3f}")
    m3.metric("Mean Precision", f"{mean_precision:.3f}")
    m4.metric("Mean Recall", f"{mean_recall:.3f}")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Radar Chart ───────────────────────────────────────────────────────────

    st.header("Per-Class mAP@50 — Radar View")
    st.markdown(
        "Each axis is one traffic sign class. A full convex polygon = perfect 1.0 mAP across all classes. "
        "The collapse at **Speed Limit 10** is the dataset's defining challenge, instantly visible here."
    )
    st.plotly_chart(plot_radar_map(metrics), use_container_width=True)

    min_class = class_names[ap50.index(min(ap50))] if ap50 else "?"
    max_class = class_names[ap50.index(max(ap50))] if ap50 else "?"
    min_val = f"{min(ap50):.3f}" if ap50 else "0.000"
    max_val = f"{max(ap50):.3f}" if ap50 else "1.000"
    st.markdown(f"""
<div class="danger-box">
<strong>Lowest mAP class:</strong> {min_class} — mAP@50 = {min_val}
</div>
<div class="insight-box">
<strong>Highest mAP class:</strong> {max_class} — mAP@50 = {max_val}
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Per-Class Bar Chart ───────────────────────────────────────────────────

    st.header("Precision / Recall / F1 per Class")
    sort_by = st.selectbox("Sort by", ["class", "F1", "mAP50"], index=0)
    st.plotly_chart(plot_per_class_bar(metrics, sort_by=sort_by), use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────

    st.header("Confusion Matrix")
    normalize = st.toggle("Normalize rows (show rates)", value=True)

    cm_data = metrics.get("confusion_matrix")
    if cm_data is not None:
        cm = np.array(cm_data)
        st.plotly_chart(
            plot_confusion_matrix(cm, class_names, normalize=normalize),
            use_container_width=True,
        )
        st.markdown("""
<div class="insight-box">
💡 <strong>Key pattern:</strong> Most off-diagonal mass clusters between visually similar speed limit
classes (e.g., 100 ↔ 110, 80 ↔ 90, 50 ↔ 60). This is expected — at distance, a single digit difference
is imperceptible. Solutions include: <em>higher resolution input</em>, <em>super-resolution pre-processing</em>,
or <em>ensemble detection + OCR verification</em>.
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Confusion matrix not found in metrics.json. Re-run train.py with the full evaluation block.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── PR Curves ─────────────────────────────────────────────────────────────

    st.header("Precision-Recall Curves")
    st.markdown("Click legend entries to isolate individual classes. Area under each curve = AP.")
    if metrics.get("pr_curves"):
        st.plotly_chart(plot_pr_curves(metrics), use_container_width=True)
    else:
        from PIL import Image as _PILImg
        _pr_png = BASE / "outputs" / "yolov8n_traffic" / "BoxPR_curve.png"
        if _pr_png.exists():
            st.image(_PILImg.open(_pr_png), use_container_width=True)
        else:
            st.info("PR curve data not available. Re-run scripts/train.py to generate it.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Confidence Violin ──────────────────────────────────────────────────────

    st.header("Confidence Score Distribution")
    st.markdown(
        "Distribution of detection confidence scores per class on the test set. "
        "Wide, high-confidence violins indicate reliable detection; "
        "narrow or low violins indicate the model is uncertain."
    )

    if CONF_CSV.exists():
        conf_df = pd.read_csv(CONF_CSV)
        st.plotly_chart(plot_violin_confidence(conf_df), use_container_width=True)
    else:
        st.info("Confidence distribution CSV not yet generated. Run `scripts/train.py` first.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Threshold Curves ──────────────────────────────────────────────────────

    from PIL import Image as _PILImg
    _CURVE_DIR = BASE / "outputs" / "yolov8n_traffic"
    _curves = [
        ("BoxF1_curve.png",  "F1 vs Confidence Threshold"),
        ("BoxPR_curve.png",  "Precision-Recall Curve"),
        ("BoxP_curve.png",   "Precision vs Confidence Threshold"),
        ("BoxR_curve.png",   "Recall vs Confidence Threshold"),
    ]
    _curve_paths = [((_CURVE_DIR / fname), title) for fname, title in _curves
                    if (_CURVE_DIR / fname).exists()]

    if _curve_paths:
        st.header("Confidence Threshold Analysis")
        st.markdown(
            "Precision, Recall, and F1 as a function of the detection confidence threshold — "
            "computed by Ultralytics across all test images. "
            "These curves let you choose the operating point that matches your safety requirements."
        )
        _cc1, _cc2 = st.columns(2)
        for _ci, (_cp, _ct) in enumerate(_curve_paths):
            _col = _cc1 if _ci % 2 == 0 else _cc2
            with _col:
                st.markdown(f"**{_ct}**")
                st.image(_PILImg.open(_cp), use_container_width=True)
        st.markdown("""
<div class="insight-box">
<strong>Safety threshold rule:</strong> For Red Light and Stop signs, read the <em>BoxR_curve</em>
and choose the threshold where <strong>Recall ≥ 0.95</strong> — even at the cost of lower precision.
A missed red light causes a collision; a false-positive brake is merely uncomfortable.
</div>
""", unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Detailed Table ────────────────────────────────────────────────────────

    st.header("Full Metrics Table")
    table_df = pd.DataFrame({
        "Class": class_names,
        "Precision": [f"{p:.3f}" for p in precision],
        "Recall": [f"{r:.3f}" for r in recall],
        "F1": [f"{f:.3f}" for f in f1_scores],
        "mAP@50": [f"{a:.3f}" for a in ap50],
        "mAP@50-95": [f"{a:.3f}" for a in ap],
    })
    st.dataframe(table_df, use_container_width=True)
    st.download_button(
        "⬇ Download metrics CSV",
        data=table_df.to_csv(index=False),
        file_name="traffic_sign_metrics.csv",
        mime="text/csv",
    )

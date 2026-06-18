import streamlit as st
import plotly.io as pio
from pathlib import Path

from utils.data_loader import BASE
from utils.visualizer import plot_loss_curves, plot_map_curves

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
pio.templates.default = "plotly_white"

RESULTS_CSV = BASE / "outputs" / "yolov8n_traffic" / "results.csv"

st.title("Model & Training")
st.markdown(
    "YOLOv8 architecture overview, training configuration, and convergence analysis. "
    "Run `python scripts/train.py` to generate the results shown here."
)

# ── Architecture ─────────────────────────────────────────────────────────────

st.header("YOLOv8 Architecture")

col_arch, col_why = st.columns(2)

with col_arch:
    st.markdown("""
```
┌──────────────────────────────────────┐
│         INPUT  416 × 416 × 3        │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│  BACKBONE  — CSPDarknet + C2f blocks │
│  • Stem Conv 3×3  →  P1 (208×208)   │
│  • C2f ×3 + SPPF  →  P3 (52×52)    │
│  • C2f ×6         →  P4 (26×26)    │
│  • C2f ×3         →  P5 (13×13)    │
│  Extracts hierarchical features at   │
│  3 scales — critical for small signs │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│  NECK  — PANet FPN (Feature Pyramid) │
│  • Up-samples P5 → fuses with P4    │
│  • Up-samples P4 → fuses with P3    │
│  • Down-samples back for rich context│
│  Enables detection at 3 scales:      │
│     P3=small, P4=medium, P5=large   │
└──────────────┬───────────────────────┘
               │
┌──────────────▼───────────────────────┐
│  HEAD  — Decoupled Detect (anchor-   │
│           free, DFL loss)            │
│  • 3 × (cls_head + reg_head)        │
│  Output: [x, y, w, h, cls_prob ×15] │
└──────────────────────────────────────┘
```
""")

with col_why:
    st.markdown("""
### Why YOLOv8 for this task?

| Requirement | YOLOv8 Answer |
|-------------|---------------|
| **Real-time** robotics deployment | 200+ FPS on GPU (nano variant) |
| **Small objects** (signs < 32px) | FPN neck fuses fine + coarse features |
| **15 classes** in one pass | Single-stage, no region proposal overhead |
| **Variable aspect ratios** | Anchor-free: no fixed anchor priors needed |
| **Edge deployment** | Export to ONNX / TensorRT / OpenVINO |

**YOLOv8n vs alternatives:**

- **vs Faster R-CNN**: 5× faster, acceptable accuracy trade-off for real-time use
- **vs SSD**: better small-object recall due to FPN
- **vs RT-DETR**: DETR needs 2× compute; YOLOv8 is more practical for edge robotics
- **vs YOLOv5**: YOLOv8 uses C2f blocks (richer gradients) + decoupled head (better cls/reg separation)

**Model variant chosen: YOLOv8n (nano)**
- 3.2M parameters, 8.7 GFLOPs
- Balances accuracy (mAP ~0.80+) with speed
- Upgrade to YOLOv8s for +3-5 mAP at 2× compute
""")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Training Config ───────────────────────────────────────────────────────────

st.header("Training Configuration")

col_t1, col_t2 = st.columns(2)

with col_t1:
    st.markdown("""
| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| `epochs` | 50 | Early stopping with patience=10 |
| `imgsz` | 416 | Matches dataset preprocessing |
| `batch` | 16 | VRAM-efficient; increase for A100 |
| `optimizer` | AdamW | Default; cosine LR decay |
| `lr0` | 0.01 | Initial learning rate |
| `lrf` | 0.01 | Final LR fraction (cosine) |
| `momentum` | 0.937 | SGD momentum equivalent |
| `weight_decay` | 0.0005 | L2 regularisation |
| `patience` | 10 | Early stopping on val mAP |
""")

with col_t2:
    st.markdown("""
| Augmentation | Value | Why |
|--------------|-------|-----|
| `mosaic` | 1.0 | Paste 4 images — context diversity |
| `mixup` | 0.1 | Soft labels for robustness |
| `degrees` | 10.0 | Rotation for sign orientation variance |
| `translate` | 0.1 | Simulate dashcam vibration |
| `scale` | 0.5 | Multi-scale training |
| `fliplr` | 0.5 | Horizontal flip (symmetric signs) |
| `hsv_h` | 0.015 | Hue shift — lighting conditions |
| `hsv_s` | 0.7 | Saturation — weather simulation |
| `hsv_v` | 0.4 | Brightness — day/night variance |
| `cls` weight | **1.5** | ↑ classification loss for imbalanced data |
""")

st.markdown("""
<div class="insight-box">
💡 <strong>Key Decision — cls=1.5:</strong> The default classification loss weight is 0.5.
Increasing it to 1.5 penalises class misidentification more heavily. Given that Speed Limit 100 and
Speed Limit 110 are visually similar (same numeral pattern, just one digit different at distance),
stronger classification supervision reduces cross-class confusion — a critical safety requirement.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Loss Curves ──────────────────────────────────────────────────────────────

st.header("Training Curves")

if RESULTS_CSV.exists():
    col_lc, col_mc = st.columns(2)
    with col_lc:
        st.plotly_chart(plot_loss_curves(RESULTS_CSV), use_container_width=True)
    with col_mc:
        st.plotly_chart(plot_map_curves(RESULTS_CSV), use_container_width=True)

    import pandas as pd
    res = pd.read_csv(RESULTS_CSV)
    res.columns = [c.strip() for c in res.columns]
    best_epoch = int(res["metrics/mAP50(B)"].idxmax()) + 1 if "metrics/mAP50(B)" in res.columns else "N/A"
    best_map50 = res["metrics/mAP50(B)"].max() if "metrics/mAP50(B)" in res.columns else 0
    best_map = res["metrics/mAP50-95(B)"].max() if "metrics/mAP50-95(B)" in res.columns else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best Epoch", best_epoch)
    m2.metric("Best mAP@50", f"{best_map50:.3f}")
    m3.metric("Best mAP@50-95", f"{best_map:.3f}")
    m4.metric("Total Epochs Run", len(res))

else:
    st.markdown("""
<div class="warning-box">
⚠️ <strong>Training results not found.</strong> Run the training script first:

```bash
cd "C:\\Users\\P Barath\\Downloads\\BENG_RMI_01"
python scripts/train.py
```

Training takes approximately:
- **GPU (RTX 3060+)**: ~20-40 minutes for 50 epochs
- **CPU only**: ~3-5 hours for 50 epochs
- **Google Colab (free T4)**: ~30-50 minutes

Once complete, reload this page to see the convergence curves.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### Expected Training Behaviour

A typical convergence curve for this dataset:

- **Epochs 1-5**: Box loss drops sharply from ~3.5 → ~1.2; model learns rough localization
- **Epochs 5-20**: Classification loss decreases as the model learns sign features
- **Epochs 20-40**: Gradual improvement; mAP@50 climbs from 0.6 → 0.80+
- **Epochs 40+**: Plateau; early stopping may trigger around epoch 45

**Expected final metrics:**
- mAP@50: ~0.80-0.85
- mAP@50-95: ~0.55-0.65
- Speed Limit 10 mAP: ~0.0 (only 19 training samples, 0 validation)
""")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Training Gallery ──────────────────────────────────────────────────────────

from PIL import Image as _PILImg

TRAIN_OUT = BASE / "outputs" / "yolov8n_traffic"

st.header("Augmented Training Samples")
st.markdown(
    "Real training batches from epoch 0 — four images composited per tile via **mosaic augmentation**. "
    "Notice varied scales, crops, and HSV colour shifts applied to maximise data diversity."
)

_train_paths = [TRAIN_OUT / f"train_batch{i}.jpg" for i in range(3)]
_train_exist = [p for p in _train_paths if p.exists()]
if _train_exist:
    _t_cols = st.columns(len(_train_exist))
    for _col, _p in zip(_t_cols, _train_exist):
        _col.image(_PILImg.open(_p), caption=_p.name, use_container_width=True)
else:
    st.info("Training batch images not found — run scripts/train.py first.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

st.header("Ground Truth vs Model Predictions")
st.markdown(
    "Left: ground-truth annotations from the validation set. "
    "Right: what the trained model predicted on the same images. "
    "Close overlap between the two indicates the model has generalised well."
)

for _i in range(3):
    _gt = TRAIN_OUT / f"val_batch{_i}_labels.jpg"
    _pr = TRAIN_OUT / f"val_batch{_i}_pred.jpg"
    if _gt.exists() and _pr.exists():
        _c_gt, _c_pr = st.columns(2)
        with _c_gt:
            st.markdown(f'<p class="video-label">Ground Truth — Batch {_i}</p>', unsafe_allow_html=True)
            st.image(_PILImg.open(_gt), use_container_width=True)
        with _c_pr:
            st.markdown(f'<p class="video-label">Model Predictions — Batch {_i}</p>', unsafe_allow_html=True)
            st.image(_PILImg.open(_pr), use_container_width=True)
        if _i < 2:
            st.markdown("---")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── DFL Loss Explainer ───────────────────────────────────────────────────────

st.header("Distribution Focal Loss — Why YOLOv8 Regresses Better")
st.markdown("""
YOLOv8 uses **Distribution Focal Loss (DFL)** for bounding box regression instead of IoU-based direct
regression. Rather than predicting a single `(x, y, w, h)`, the model outputs a **probability distribution
over possible coordinate values**, then takes the expectation.

```
Traditional regression:  predict Δx directly  → one point estimate, high uncertainty
DFL approach:            predict P(Δx = k) for k ∈ {0,...,15}
                         → take E[Δx] = Σ k · P(k)
```

**Why this matters for traffic signs:**
- Signs at distance have ambiguous boundaries → DFL captures this uncertainty in the distribution spread
- Sign boundary detection is sharper → higher IoU with ground truth
- Results in higher mAP@50-95 (strict IoU thresholds) compared to traditional regression
""")

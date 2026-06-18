from __future__ import annotations
import streamlit as st
import plotly.io as pio
from pathlib import Path

css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
pio.templates.default = "plotly_white"

st.title("Insights")
st.markdown("Key findings from the data analysis, model evaluation, and identified challenges.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Key Findings ──────────────────────────────────────────────────────────────

st.header("Key Findings")

st.markdown("""
<div class="danger-box">
<strong>Finding 1: Severe class imbalance</strong><br/>
Speed Limit 10 has only <strong>19 training samples</strong> and zero validation samples, which is a 30x
imbalance compared to Red Light (585 samples). The model cannot learn this class reliably,
so its mAP comes out near zero. This is a real data distribution problem, not a labelling mistake.
Speed Limit 10 zones are just rare in dashcam footage.
</div>

<div class="insight-box">
<strong>Finding 2: Signs cluster in the upper-center of the frame</strong><br/>
The spatial heatmap shows roughly 80% of annotations land in the upper half and central 60% of frame
width. This makes sense for a forward-facing dashcam. It also means you could crop just that region
before running the detector and save a significant amount of compute.
</div>

<div class="warning-box">
<strong>Finding 3: Most signs are small objects</strong><br/>
Median bounding box size is about 28x28 px at 416-px resolution, which puts them firmly in the
"small object" category by COCO standards. This is exactly why the FPN neck in YOLOv8 matters:
it brings back fine spatial detail from early layers that deep layers would otherwise throw away.
</div>

<div class="warning-box">
<strong>Finding 4: Visually similar speed limit classes cause confusion</strong><br/>
Speed Limit 100 vs 110 vs 120 differ by just one digit, and at distance that digit might only span
3 to 5 pixels. The confusion matrix clearly shows most misclassifications happen between these
adjacent classes. A two-stage approach (detect the sign first, then run OCR on the crop) would
handle this much better.
</div>

<div class="insight-box">
<strong>Finding 5: No augmentation in the source dataset</strong><br/>
The original Roboflow export had no augmentation applied to it. The training pipeline adds mosaic,
mixup, HSV shifts and rotation to make up for that, but the base distribution still does not cover
rain, fog, night or motion blur conditions.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Perception Pipeline ───────────────────────────────────────────────────────

st.header("Where Detection Fits in the Driving Pipeline")

col_pipe, col_explain = st.columns(2)

with col_pipe:
    st.markdown("""
```
SENSOR LAYER
  Camera -> Raw video frames

         |
         v

PERCEPTION LAYER   <- This project
  YOLOv8 detector
  Output: {class, confidence, bbox}

         |
         v

PLANNING LAYER
  Red Light    -> decelerate -> stop
  Speed Limit  -> cap velocity
  Stop Sign    -> full stop 3 s
  Green Light  -> proceed

         |
         v

CONTROL LAYER
  Throttle / brake / steering
```
""")

with col_explain:
    st.markdown("""
### Detection to decision mapping

```python
for det in detections:
    if det["class"] == "Red Light":
        planner.set_velocity(0)

    elif "Speed Limit" in det["class"]:
        limit = int(det["class"].split()[-1])
        planner.set_max_velocity(limit / 3.6)

    elif det["class"] == "Stop":
        planner.full_stop(duration=3)
```

**On confidence thresholds and safety:**
Red lights and stop signs use a lower threshold (higher recall)
because a missed detection is more dangerous than a false positive.
""")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Challenges ────────────────────────────────────────────────────────────────

st.header("Engineering Challenges")

tab1, tab2, tab3 = st.tabs([
    "Class Imbalance", "Small Objects", "Visual Similarity"
])

with tab1:
    st.markdown("""
### Long-Tail Distribution

The dataset has a big imbalance problem. Red Light and Green Light each have 500+ training samples,
but Speed Limit 10 only has 19. The model barely sees Speed Limit 10 during training, so it never
learns to detect it properly. The final mAP for that class is near zero.

This is not a bug in the dataset. It just reflects how rarely Speed Limit 10 zones appear in
real dashcam footage.

**What can be done about it:**

1. **Copy-paste augmentation**: Take the 19 existing Speed Limit 10 crops and paste them
   into other training images at different sizes and positions. This is cheap and effective.

2. **Increase the classification loss weight**: The training here already uses cls=1.5 instead
   of the default 0.5. This makes the model pay more attention to getting the class right,
   not just the bounding box location.

3. **Few-shot learning**: Train a base model on the majority classes first, then fine-tune it
   on the rare classes with just a small number of examples. Techniques like prototypical
   networks or MAML are designed for exactly this.

4. **Synthetic data**: Use a driving simulator like CARLA or Blender to render Speed Limit 10
   signs under different lighting and camera angles. It is not perfect, but it fills the gap.
""")

with tab2:
    st.markdown("""
### Small Object Detection

The median sign size in this dataset is about 28x28 pixels at 416-px resolution. That is less
than 0.5% of the total image area. Small objects are genuinely hard for CNNs to detect.

**Why it is hard:**

Deep layers in a CNN produce small feature maps like 13x13 or 26x26. A 28-pixel object in a
416-pixel image only takes up about 3 cells in a 13x13 map, which does not give the network
much to work with. Fine spatial details get lost as you go deeper.

**How YOLOv8 handles it with the Feature Pyramid Network:**

```
P5 (13x13) -> detect large objects
P4 (26x26) -> detect medium objects
P3 (52x52) -> detect small signs   <-- this is the key level
```

The FPN passes feature information back up from deep layers to shallow layers, so the P3 output
has both fine spatial detail and high-level semantic context. Without this, small signs would be
nearly undetectable.

**What could improve it further:**
- Increase input resolution to 640x640 (gives roughly +5 to 8 mAP on small classes, costs 2x compute)
- SAHI (Slicing Aided Hyper Inference): split the image into tiles, run inference on each tile
  separately, then merge the results back
""")

with tab3:
    st.markdown("""
### Visually Similar Classes

Speed limit signs all look the same from a distance: round, red border, white background, black
number. At 10 to 30 metres, a sign might only be 20 to 40 pixels wide. At that size, the
difference between "100" and "110" is maybe two or three pixels. The model often gets confused.

**Worst confusion pairs from the confusion matrix:**
- Speed Limit 100 vs 110 (one extra digit)
- Speed Limit 80 vs 90 (visually similar numeral shapes)
- Speed Limit 50 vs 60 (similar at low resolution)

**What can be done:**

1. **Higher input resolution**: Running at 640x640 instead of 416x416 gives the model more
   pixels per sign, which makes digit separation more feasible.

2. **Two-stage detection with OCR**: Stage 1 just detects "speed limit sign" as a single class.
   Stage 2 crops that region and runs a text recognition model like TrOCR or PaddleOCR to
   read the actual number. This sidesteps the whole classification confusion problem.

3. **Temporal consistency**: In a video stream, aggregate predictions over 5 consecutive frames
   and use majority voting. If the model says "60" for one frame but "50" for the surrounding
   four frames, trust the majority. This removes single-frame noise without adding much latency.
""")


st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Future Work ───────────────────────────────────────────────────────────────

st.header("Future Improvements")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
**Near-term**
- Collect 200+ Speed Limit 10 images from GTSRB or Mapillary to fix the imbalance
- Switch input resolution to 640x640
- Add temporal consistency with a 5-frame majority vote
- Apply temperature scaling to calibrate confidence scores
""")
with col_b:
    st.markdown("""
**Medium-term**
- Cascaded detection + OCR for all speed limit classes
- Sensor fusion with LiDAR for 3D sign localisation
- Synthetic rain and night data from CARLA simulator
- Deploy as a ROS2 perception node publishing DetectionArray messages
""")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.caption("Dataset: Roboflow Self-Driving Cars Traffic Signs · CC BY 4.0")

"""Adds markdown cells (before AND after code cells) to the training notebook.
Strips any markdown cells added by a previous run first, then re-inserts fresh ones.
No em dashes used anywhere.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "Traffic_Signs_ComputerVision_Project.ipynb"

with open(NB_PATH, encoding="utf-8") as f:
    nb = json.load(f)


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip(),
    }


# Strip any markdown cells from a previous run so we start clean
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
print(f"Found {len(code_cells)} code cells to work with.")


# =============================================================================
# PRE-CELL markdowns  (explain what the cell is about to do)
# =============================================================================

INTRO = md("""# Traffic Sign Detection: YOLOv8 Training on Google Colab

**Dataset:** Roboflow Self-Driving Cars Traffic Signs (CC BY 4.0)
**Model:** YOLOv8n fine-tuned from COCO pretrained weights
**Classes:** 15 traffic sign types across 4,969 dashcam images
**Final result:** mAP@50 = 0.932 on the held-out test set

This notebook trains a YOLOv8 object detection model on a real dashcam traffic sign dataset,
evaluates it on a held-out test set, runs inference on a dashcam video, and packages all
outputs into a zip file for use in a local Streamlit dashboard.

Run all cells from top to bottom. Each cell is explained before and after its output.
""")

PRE = [
    # Cell 0: GPU check
    md("""## Step 1: Check GPU

Training YOLOv8 on CPU takes 3 to 5 hours for 50 epochs. On a Tesla T4 GPU it takes around
30 to 40 minutes. Check here before doing anything else.

If you see "No GPU detected", go to **Runtime > Change runtime type** and pick **T4 GPU**,
then re-run from the top.
"""),

    # Cell 1: Mount Drive
    md("""## Step 2: Mount Google Drive

The dataset is sitting in your Google Drive. Mounting it makes it accessible at
`/content/drive/MyDrive/`.

The dataset folder must be named exactly: `Traffic_Signs_Detection_ComputerVision`
"""),

    # Cell 2: Install deps
    md("""## Step 3: Install Dependencies

Installs `ultralytics`, which bundles YOLOv8, PyTorch, and everything needed to train
and evaluate an object detection model. `opencv-python-headless` is the server version
of OpenCV that works without a display.

This takes around 30 seconds.
"""),

    # Cell 3: Copy dataset
    md("""## Step 4: Copy Dataset to Local Storage

Google Drive reads are slow when processing thousands of small image files. During
training, YOLOv8 reads every image multiple times per epoch. Doing that over Drive
is a bottleneck that slows training by 3 to 4 times.

Copying the full dataset (around 500 MB) to Colab's local disk once at the start
fixes this. All subsequent reads during training hit the fast local SSD.
"""),

    # Cell 4: Patch yaml
    md("""## Step 5: Patch data.yaml with Absolute Paths

The `data.yaml` from Roboflow uses relative paths. These do not resolve correctly
in Colab's filesystem. This cell rewrites the train, val, and test paths to absolute
paths pointing to the locally copied dataset.

The 15 class names are printed as a sanity check to confirm everything loaded correctly.
"""),

    # Cell 5: Train
    md("""## Step 6: Train YOLOv8n (around 30 to 40 minutes on T4)

Fine-tunes a YOLOv8 nano model starting from COCO pretrained weights. Key parameter decisions:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `epochs` | 50 | Upper limit; early stopping handles the actual cutoff |
| `patience` | 10 | Stops training if val mAP does not improve for 10 consecutive epochs |
| `imgsz` | 416 | Matches the dataset native resolution |
| `batch` | 16 | Fits T4 VRAM comfortably |
| `cls` | 1.5 | Default is 0.5. Increased to penalise misclassification harder. Speed limit signs look visually similar, so getting the class right matters more than a slightly imprecise box |
| `mosaic` | 1.0 | Pastes 4 images together per training tile, increasing scale and context variety |
| `mixup` | 0.1 | Blends two images with soft labels to improve generalisation |
| `degrees` | 10.0 | Small rotation to simulate camera angle variation |

The best checkpoint (by validation mAP@50) is saved automatically to `weights/best.pt`.
"""),

    # Cell 6: Export metrics
    md("""## Step 7: Evaluate on Test Set and Export Metrics

Loads the best checkpoint and runs it on 638 test images that the model never saw during
training or validation. This is the honest evaluation.

Per-class precision, recall, mAP@50, and mAP@50-95 are saved to `metrics.json` for
the Streamlit dashboard.
"""),

    # Cell 7: Confidence CSV
    md("""## Step 8: Export Confidence Score Distributions

Runs the trained model on 400 test images with a low confidence threshold of 0.1.
The low threshold captures borderline detections as well, not just confident ones.
This gives a full distribution of confidence scores per class.

The Streamlit dashboard uses this CSV to draw violin plots showing how certain the model
is about each sign type. Wide, high-confidence violins mean the model is reliable for
that class. Sparse or low-score violins signal uncertainty, usually from limited training
data or visual similarity with other classes.
"""),

    # Cell 8: Video
    md("""## Step 9: Run Inference on the Dashcam Video

Processes every frame of the 508-frame dashcam video (30 FPS, around 17 seconds).
Detected signs get coloured bounding boxes and class labels drawn on each frame.
The annotated output is saved as `processed_video.mp4`.

Per-frame detection counts and per-class totals are saved to `video_stats.json`
for the dashboard timeline chart.
"""),

    # Cell 9: Save to Drive
    md("""## Step 10: Package and Save Outputs to Google Drive

Zips the entire `/content/outputs/` folder and copies it to your Drive as
`BENG_RMI_outputs.zip`. This is the file you download and extract locally
to run the Streamlit dashboard.

The zip contains model weights, metrics, training visualisations, processed video,
confidence CSVs, and per-frame stats.
"""),

    # Cell 10: Download
    md("""## Step 11: Download Zip to Your Computer

Downloads the zip directly to your local machine from Colab.

Once downloaded:
1. Extract the zip file
2. Place the `outputs/` folder inside your `BENG_RMI_01/` project directory
3. Run `streamlit run app.py` from that directory

The dashboard will detect all outputs automatically and load every chart and the
inference demo without any extra setup.
"""),
]


# =============================================================================
# POST-CELL markdowns  (explain what the output means)
# =============================================================================

POST = [
    # After Cell 0: GPU result
    md("""The Tesla T4 has 15 GB of VRAM and delivers around 8.1 TFLOPS of FP16 compute.
This is more than enough for training YOLOv8n with batch=16 at 416 px. The GPU shows
0% utilisation and 9W power draw right now because nothing is running yet. That is
expected and correct.
"""),

    # After Cell 1: Mount Drive
    None,  # straightforward, no explanation needed

    # After Cell 2: Install deps
    md("""Ultralytics 8.4.69 is the version used throughout this project. This version
includes the DFL (Distribution Focal Loss) regression head which improves bounding box
precision compared to earlier YOLO versions, and the C2f backbone blocks that give
richer gradient flow than C3 blocks used in YOLOv5.
"""),

    # After Cell 3: Copy dataset
    md("""3,530 training images confirmed. The full split is:
- Train: 3,530 images
- Validation: 801 images
- Test: 638 images (held out, not used until evaluation)

The dashcam video is also confirmed present. If either number looks wrong, re-check
the folder name in Drive before continuing.
"""),

    # After Cell 4: Patch yaml
    md("""All 15 classes loaded correctly. The class ordering here defines the numeric class IDs
used in all YOLO label files: 0 = Green Light, 1 = Red Light, 2 = Speed Limit 10, and
so on. This ordering must stay consistent between training and inference.

Worth noting: Speed Limit 10 is class ID 2. Despite being present in the dataset, it
only has 19 training samples compared to 585 for Red Light. That is a 30x imbalance
and it directly causes Speed Limit 10 to have the lowest detection accuracy of all classes.
"""),

    # After Cell 5: Train (output too large but explain what to look for)
    md("""**What to look at in the training output above:**

The table printed each epoch has these key columns:
- `box_loss`: how well the model places bounding boxes. Should decrease each epoch.
- `cls_loss`: how well the model identifies the sign class. Also decreases, but more slowly
  since classification is harder than localisation.
- `mAP50`: validation mAP at IoU 0.5. This is the number to watch. It should rise steadily
  and then flatten out when early stopping triggers.

If training stopped before epoch 50, early stopping kicked in because val mAP did not
improve for 10 consecutive epochs. That is normal and means the model had already converged.

The best checkpoint is saved when val mAP peaks, not at the final epoch.
"""),

    # After Cell 6: Metrics export
    md("""**Result: mAP@50 = 0.9316, mAP@50-95 = 0.7670**

What these numbers mean in practice:

- **mAP@50 = 0.932**: Across all 15 classes, the model correctly detects and identifies
  traffic signs with at least 50% bounding box overlap 93.2% of the time. This is a strong
  result for a nano-size model on a real-world dataset.

- **mAP@50-95 = 0.767**: The stricter metric that averages mAP across IoU thresholds from
  0.50 to 0.95. This measures how precisely the boxes are drawn, not just whether the
  detection is roughly correct. The gap between 0.932 and 0.767 is normal and shows the
  model is accurate at detection but the box edges are not always pixel-perfect.

- **Inference speed: 2.3ms per image**: That is roughly 430 frames per second. A dashcam
  runs at 30 FPS, so this model has about 14x more compute headroom than it needs. Even
  on less powerful hardware, real-time deployment is feasible.

- **Precision = 0.864, Recall = 0.886**: Recall is slightly higher than precision.
  The model misses fewer signs than it falsely triggers on, which is the right trade-off
  for a safety application.
"""),

    # After Cell 7: Confidence CSV
    md("""563 detections collected across 400 test images. That is about 1.4 detections per
image on average, which makes sense since most dashcam frames contain one or two signs
at most.

The low threshold of 0.1 means borderline detections are included. These low-confidence
cases are what make the violin plots in the dashboard informative. A class with all its
detections bunched above 0.85 is reliable. A class with detections spread from 0.1 to
0.9 is one the model is uncertain about.
"""),

    # After Cell 8: Video
    md("""**508 frames processed at 30 FPS.**

Detection breakdown:

| Class | Count | What this tells us |
|-------|-------|--------------------|
| Green Light | 238 | The road passes through multiple signalised intersections |
| Red Light | 156 | About 1 in 3 green light frames also has a red light nearby |
| Speed Limit 50 | 65 | Most of the footage is in a 50 km/h zone |
| Speed Limit 90 | 30 | A short highway section in the video |
| Speed Limit 20 | 54 | School zone or residential street section |
| Speed Limit 10 | 8 | Despite having only 19 training samples, the model still detected this class 8 times in real footage. It learned something, just not enough to be reliable. |

The `detections_per_frame` list in `video_stats.json` is what the dashboard uses to
draw the timeline chart showing detection activity across the video.
"""),

    # After Cell 9: Save to Drive
    md("""The zip is saved to your Drive at `MyDrive/BENG_RMI_outputs.zip`.

File sizes worth noting:
- `best.pt` at 5.9 MB is the full trained model. This is small enough to deploy on
  embedded hardware like a Jetson Nano or a Raspberry Pi 5 with a USB camera.
- `processed_video.mp4` at 3.9 MB is the annotated dashcam video. The original video
  is about the same size, so the detection overlay adds minimal overhead.
- Total outputs excluding weights: under 5 MB. The full zip including weights is around
  22 MB.
"""),

    # After Cell 10: Download
    md("""The download starts automatically in your browser. The file is named
`outputs_for_download.zip`.

After extracting, the folder structure inside should match what the Streamlit dashboard
expects. Place everything into `BENG_RMI_01/outputs/` and run:

```bash
streamlit run app.py
```

No retraining needed. The dashboard reads the weights, metrics, and videos directly
from the outputs folder.
"""),
]


# =============================================================================
# Rebuild notebook: INTRO + (pre, code, post) for each cell
# =============================================================================

new_cells = [INTRO]

for i, code_cell in enumerate(code_cells):
    if i < len(PRE):
        new_cells.append(PRE[i])
    new_cells.append(code_cell)
    if i < len(POST) and POST[i] is not None:
        new_cells.append(POST[i])

nb["cells"] = new_cells

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

md_count = sum(1 for c in new_cells if c["cell_type"] == "markdown")
code_count = sum(1 for c in new_cells if c["cell_type"] == "code")
print(f"Done. {len(new_cells)} total cells: {code_count} code + {md_count} markdown.")

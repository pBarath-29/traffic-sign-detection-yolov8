# Traffic Sign Detection: YOLOv8 Computer Vision Pipeline

A full end-to-end object detection project trained on 4,969 real dashcam images across 15 traffic sign classes, achieving **93.2% mAP@50** on a held-out test set. Built with YOLOv8, Streamlit, and Python.

**Live demo:** [traffic-sign-detection-yolov8.streamlit.app](https://traffic-sign-detection-yolov8-sapjre6eunhvzgdt7dxcyb.streamlit.app/)

---

## What this project does

A vehicle driving on a public road needs to recognise traffic signs fast and reliably. This project builds and evaluates a vision-based perception module that does exactly that: given a dashcam image or video frame, it detects every traffic sign in the scene, labels its class, outputs a confidence score, and estimates the sign's distance from the camera.

The pipeline covers everything from raw dataset analysis to live inference on video. The dataset analysis stage produces class distribution plots, spatial heatmaps, and bounding box statistics. Model training fine-tunes YOLOv8n with augmentation and imbalance-aware loss. Evaluation reports per-class mAP, precision-recall curves, and a confusion matrix. The inference stage provides a before/after dashcam video comparison and a live image upload tool. An explainability layer adds EigenCAM neural attention maps and distance estimation.

---

## Results

Evaluated on 638 test images never seen during training or validation:

| Metric | Value |
|--------|-------|
| mAP@50 | **0.932** |
| mAP@50-95 | **0.767** |
| Mean Precision | 0.872 |
| Mean Recall | 0.876 |

The best performing classes were Stop (mAP 0.995), Speed Limit 100 (0.985), and Speed Limit 120 (0.987). The hardest class was Speed Limit 10 (mAP 0.830), which had only 19 training samples in the entire dataset, a 30x imbalance relative to Red Light.

---

## Dataset

The dataset is sourced from Roboflow Self-Driving Cars Traffic Signs (CC BY 4.0) and consists of 4,969 real dashcam frames at 416x416 px, annotated with 6,012 bounding boxes in YOLO label format. The split is 3,530 train, 801 validation, and 638 test images across 15 classes: Green Light, Red Light, Stop, and Speed Limit 10/20/30/40/50/60/70/80/90/100/110/120.

The dataset has a severe long-tail distribution. Red Light has 585 training samples while Speed Limit 10 has only 19. This is not a labelling problem; it reflects how rarely 10 km/h zones appear in real dashcam footage. Handling this kind of imbalance is one of the core engineering challenges explored in this project.

---

## Technical choices

**Why YOLOv8?**

YOLOv8 is a single-stage detector, which means it classifies and localises objects in one forward pass. The Feature Pyramid Network neck fuses features at three scales (13x13, 26x26, 52x52), which is essential because most signs in this dataset are only 28x28 px at 416-px resolution. The anchor-free head avoids having to tune anchor sizes per dataset. The nano variant (YOLOv8n, 3.2M parameters) runs at 200+ FPS on a GPU, which matters if you are integrating this into a real-time vehicle system. Exporting to ONNX or TensorRT for edge deployment is also straightforward.

**Training decisions**

The default classification loss weight in YOLOv8 is 0.5. This project increases it to 1.5 (`cls=1.5`) because the most common failure mode here is not missing a sign but misidentifying it, such as confusing Speed Limit 100 with 110, or 80 with 90. Penalising classification errors more heavily partially compensates for this. Mosaic and mixup augmentation are both enabled. Mosaic composites four training images into one tile, which effectively quadruples the variety of sign scales and contexts the model sees per batch.

---

## Explainability

Beyond mAP numbers, this project includes two features that explain what the model is actually doing.

**EigenCAM attention maps:** When you upload an image on the Inference page, the app shows which regions of the image the YOLOv8 backbone focused on. The target layer is the SPPF bottleneck at depth 9 in the backbone. Hot areas in the heatmap correspond to the spatial features that drove the predictions.

**Distance estimation:** For each detected sign, the app estimates distance using the pinhole camera model:

```
distance (m) = (real_sign_height_m x focal_length_px) / bbox_height_px
```

Standard EU traffic sign heights are used (0.6 m for speed limit signs, 0.8 m for traffic lights). Focal length is assumed at 416 px, consistent with a 90-degree HFOV dashcam. This links the detection output to a physically meaningful quantity that a planning module can use.

---

## Where this fits in a robotics stack

Traffic sign detection is a perception task. In a full autonomous driving system, perception sits between the sensor layer and the planning layer:

```
Camera
  -> Perception (this project: detect and classify signs)
  -> Planning   (decide: stop, slow down, proceed)
  -> Control    (execute: throttle, brake, steering)
```

The detection output from this model maps directly to planner inputs:

```python
if class == "Red Light":
    planner.set_velocity(0)
elif "Speed Limit" in class:
    planner.set_max_velocity(int(limit) / 3.6)
elif class == "Stop":
    planner.full_stop(duration=3)
```

A natural next step would be to deploy this as a ROS2 node that publishes a `DetectionArray` message on each camera frame, so the planner and controller can subscribe to it directly.

---

## Project structure

```
BENG_RMI_01/
├── app.py                      # Streamlit landing page
├── pages/
│   ├── 1_Dataset_Explorer.py   # EDA: distributions, heatmaps, mosaic
│   ├── 2_Model_Training.py     # Architecture, config, training curves, GT vs predictions
│   ├── 3_Performance.py        # mAP, radar chart, confusion matrix, PR curves
│   ├── 4_Inference_Demo.py     # Video demo, live upload, GradCAM, distance estimation
│   └── 5_Insights.py           # Key findings, challenges, future work
├── utils/
│   ├── data_loader.py          # Dataset parsing and image utilities
│   ├── visualizer.py           # All Plotly chart functions
│   └── detector.py             # YOLOv8 inference, GradCAM, distance estimation
├── scripts/
│   ├── train.py                # Training script (run once)
│   └── process_video.py        # Video inference (run after training)
├── assets/
│   └── style.css
└── outputs/                    # Generated: weights, metrics, processed video
```

---

## Dependencies

```
ultralytics       # YOLOv8
streamlit         # Dashboard
plotly            # Interactive charts
opencv-python     # Video processing
pytorch-grad-cam  # EigenCAM explainability
numpy, pandas, Pillow, pyyaml
```

---

## Known limitations

Speed Limit 10 has near-zero training data (19 samples), so the model detects it unreliably. Collecting more data or using synthetic augmentation would address this. Distance estimates assume a fixed focal length; a camera calibration step would make these more accurate. Speed limit classes 100, 110, and 120 are frequently confused because the digit difference is only a few pixels at typical detection distances, and a two-stage detection plus OCR approach would handle this better.

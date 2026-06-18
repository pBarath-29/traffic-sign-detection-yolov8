"""All visualization functions. Each returns a Plotly Figure or numpy array.
Never calls st.* directly — display logic lives in page files."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import CLASS_COLORS, get_class_names, load_image_with_boxes

GRID = "#f3f4f6"
FONT_COLOR = "#111827"
LEGEND_BG  = "rgba(255,255,255,0.9)"


def class_color_map() -> Dict[str, str]:
    names = get_class_names()
    return {name: CLASS_COLORS[i % len(CLASS_COLORS)] for i, name in enumerate(names)}


# ── EDA Charts ───────────────────────────────────────────────────────────────

def plot_class_distribution(counts_df: pd.DataFrame) -> go.Figure:
    split_colors = {"train": "#2563eb", "valid": "#16a34a", "test": "#d97706"}
    fig = go.Figure()
    for split in ["train", "valid", "test"]:
        subset = counts_df[counts_df["split"] == split].sort_values("class_id")
        fig.add_trace(go.Bar(
            name=split.capitalize(),
            x=subset["class_name"],
            y=subset["count"],
            marker_color=split_colors.get(split, "#9ca3af"),
            marker_line_width=0,
            hovertemplate="<b>%{x}</b><br>Split: " + split + "<br>Count: %{y}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="Annotation Count per Class and Split", font=dict(size=15, color=FONT_COLOR)),
        xaxis=dict(title="Traffic Sign Class", tickangle=-35, tickfont=dict(size=11), gridcolor=GRID),
        yaxis=dict(title="Annotation Count", gridcolor=GRID),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor=LEGEND_BG),
        height=420,
        margin=dict(b=80),
    )
    return fig


def plot_spatial_heatmap(density: np.ndarray, title_suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=density,
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Density", tickfont=dict(color=FONT_COLOR)),
        hovertemplate="Col: %{x}<br>Row: %{y}<br>Density: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=f"Bounding Box Spatial Density{' — ' + title_suffix if title_suffix else ''}",
            font=dict(size=15, color=FONT_COLOR),
        ),
        xaxis=dict(title="Image Width →", showgrid=False, zeroline=False),
        yaxis=dict(title="Image Height ↓", showgrid=False, zeroline=False, autorange="reversed"),
        height=400,
    )
    return fig


def plot_bbox_size_scatter(df: pd.DataFrame) -> go.Figure:
    color_map = class_color_map()
    fig = go.Figure()
    for class_name, group in df.groupby("class_name"):
        fig.add_trace(go.Scatter(
            x=group["w"] * 416,
            y=group["h"] * 416,
            mode="markers",
            name=class_name,
            marker=dict(color=color_map.get(class_name, "#888"), size=4, opacity=0.5),
            hovertemplate=f"<b>{class_name}</b><br>W: %{{x:.1f}}px<br>H: %{{y:.1f}}px<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Bounding Box Size Distribution (px)", font=dict(size=15, color=FONT_COLOR)),
        xaxis=dict(title="Width (px)", gridcolor=GRID),
        yaxis=dict(title="Height (px)", gridcolor=GRID),
        height=420,
        legend=dict(font=dict(size=9), itemsizing="constant", bgcolor=LEGEND_BG,
                    bordercolor="#e5e7eb"),
    )
    return fig


def plot_donut_imbalance(counts_df: pd.DataFrame, split: str = "train") -> go.Figure:
    subset = counts_df[counts_df["split"] == split].sort_values("class_id")
    colors = [CLASS_COLORS[i % len(CLASS_COLORS)] for i in subset["class_id"]]
    min_idx = subset["count"].idxmin()
    colors[subset.index.get_loc(min_idx)] = "#dc2626"
    total = subset["count"].sum()
    fig = go.Figure(go.Pie(
        labels=subset["class_name"],
        values=subset["count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Class Distribution — {split.capitalize()} Split",
                   font=dict(size=15, color=FONT_COLOR)),
        annotations=[dict(text=f"<b>{total:,}</b><br>total", x=0.5, y=0.5,
                          font_size=13, showarrow=False, font_color=FONT_COLOR)],
        height=420,
        showlegend=False,
    )
    return fig


def plot_mosaic(stems: List[str], split: str, df: pd.DataFrame,
                cols: int = 5, img_size: int = 200) -> np.ndarray:
    from PIL import Image as _PILImage
    rows_count = max(1, math.ceil(len(stems) / cols))
    canvas = np.full((rows_count * img_size, cols * img_size, 3), 240, dtype=np.uint8)
    for idx, stem in enumerate(stems):
        r, c = idx // cols, idx % cols
        img = load_image_with_boxes(stem, split, df, img_size=img_size)
        if img.shape[0] != img_size or img.shape[1] != img_size:
            img = np.array(_PILImage.fromarray(img).resize((img_size, img_size)))
        canvas[r*img_size:(r+1)*img_size, c*img_size:(c+1)*img_size] = img
    return canvas


# ── Training Charts ───────────────────────────────────────────────────────────

def plot_loss_curves(csv_path: Path) -> go.Figure:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Box Loss", "Classification Loss"))
    for (train_col, val_col, col) in [
        ("train/box_loss", "val/box_loss", 1),
        ("train/cls_loss", "val/cls_loss", 2),
    ]:
        if train_col in df.columns:
            fig.add_trace(go.Scatter(x=df.index+1, y=df[train_col],
                name=f"Train", line=dict(color="#2563eb", width=2), mode="lines"), row=1, col=col)
        if val_col in df.columns:
            fig.add_trace(go.Scatter(x=df.index+1, y=df[val_col],
                name=f"Val", line=dict(color="#dc2626", width=2, dash="dash"), mode="lines"), row=1, col=col)
    fig.update_layout(
        title=dict(text="Training Loss Curves", font=dict(size=15, color=FONT_COLOR)),
        height=350,
        legend=dict(bgcolor=LEGEND_BG, bordercolor="#e5e7eb"),
    )
    fig.update_xaxes(title_text="Epoch", gridcolor=GRID)
    fig.update_yaxes(title_text="Loss", gridcolor=GRID)
    return fig


def plot_map_curves(csv_path: Path) -> go.Figure:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    fig = go.Figure()
    if "metrics/mAP50(B)" in df.columns:
        fig.add_trace(go.Scatter(x=df.index+1, y=df["metrics/mAP50(B)"],
            name="mAP@50", line=dict(color="#2563eb", width=2.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.07)"))
    if "metrics/mAP50-95(B)" in df.columns:
        fig.add_trace(go.Scatter(x=df.index+1, y=df["metrics/mAP50-95(B)"],
            name="mAP@50-95", line=dict(color="#16a34a", width=2.5),
            fill="tozeroy", fillcolor="rgba(22,163,74,0.06)"))
    fig.update_layout(
        title=dict(text="mAP Progress Over Training", font=dict(size=15, color=FONT_COLOR)),
        xaxis=dict(title="Epoch", gridcolor=GRID),
        yaxis=dict(title="mAP", gridcolor=GRID, range=[0, 1]),
        height=350,
        legend=dict(bgcolor=LEGEND_BG, bordercolor="#e5e7eb"),
    )
    return fig


# ── Performance Charts ────────────────────────────────────────────────────────

def plot_radar_map(metrics: dict) -> go.Figure:
    class_names = metrics.get("class_names", [])
    ap50 = metrics.get("ap50", [])
    if not class_names or not ap50:
        return go.Figure()
    theta = class_names + [class_names[0]]
    r = list(ap50) + [ap50[0]]
    fig = go.Figure(go.Scatterpolar(
        r=r, theta=theta, fill="toself",
        fillcolor="rgba(37,99,235,0.12)",
        line=dict(color="#2563eb", width=2),
        hovertemplate="<b>%{theta}</b><br>mAP@50: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Per-Class mAP@50", font=dict(size=15, color=FONT_COLOR)),
        polar=dict(
            bgcolor="#f9fafb",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#e5e7eb",
                            tickfont=dict(color="#6b7280", size=9)),
            angularaxis=dict(gridcolor="#e5e7eb", tickfont=dict(color="#374151", size=10)),
        ),
        height=480,
    )
    return fig


def plot_per_class_bar(metrics: dict, sort_by: str = "class") -> go.Figure:
    class_names = metrics.get("class_names", [])
    precision   = metrics.get("precision", [])
    recall      = metrics.get("recall", [])
    ap50        = metrics.get("ap50", [])
    f1 = [2*p*r/(p+r+1e-8) for p, r in zip(precision, recall)]
    df = pd.DataFrame({"class": class_names, "Precision": precision,
                       "Recall": recall, "F1": f1, "mAP50": ap50})
    if sort_by == "F1":       df = df.sort_values("F1", ascending=False)
    elif sort_by == "mAP50":  df = df.sort_values("mAP50", ascending=False)
    fig = go.Figure()
    for metric, color in [("Precision","#2563eb"), ("Recall","#16a34a"), ("F1","#d97706")]:
        fig.add_trace(go.Bar(name=metric, x=df["class"], y=df[metric],
            marker_color=color, opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.3f}}<extra></extra>"))
    fig.update_layout(
        barmode="group",
        title=dict(text="Per-Class Precision / Recall / F1", font=dict(size=15, color=FONT_COLOR)),
        xaxis=dict(title="Class", tickangle=-40, tickfont=dict(size=10), gridcolor=GRID),
        yaxis=dict(title="Score", range=[0, 1.05], gridcolor=GRID),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(b=80),
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          normalize: bool = True) -> go.Figure:
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        data = (cm / row_sums).round(3)
        fmt = ".2f"
    else:
        data = cm
        fmt = "d"
    text_matrix = [[f"{v:{fmt}}" for v in row] for row in data]
    fig = go.Figure(go.Heatmap(
        z=data, x=class_names, y=class_names,
        text=text_matrix, texttemplate="%{text}",
        colorscale="RdYlGn", zmin=0, zmax=1 if normalize else None,
        colorbar=dict(title="Rate" if normalize else "Count"),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Value: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Confusion Matrix {'(Normalized)' if normalize else '(Raw Counts)'}",
                   font=dict(size=15, color=FONT_COLOR)),
        xaxis=dict(title="Predicted", tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(title="Actual", tickfont=dict(size=9), autorange="reversed"),
        height=560, margin=dict(b=100, l=100),
    )
    return fig


def plot_pr_curves(metrics: dict) -> go.Figure:
    class_names = metrics.get("class_names", [])
    pr_data     = metrics.get("pr_curves", {})
    fig = go.Figure()
    for i, name in enumerate(class_names):
        if name in pr_data:
            curve = pr_data[name]
            fig.add_trace(go.Scatter(
                x=curve.get("recall", []), y=curve.get("precision", []),
                name=name, mode="lines",
                line=dict(color=CLASS_COLORS[i % len(CLASS_COLORS)], width=2),
                hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>",
            ))
    fig.update_layout(
        title=dict(text="Precision-Recall Curves (per class)", font=dict(size=15, color=FONT_COLOR)),
        xaxis=dict(title="Recall", range=[0, 1], gridcolor=GRID),
        yaxis=dict(title="Precision", range=[0, 1.05], gridcolor=GRID),
        height=440,
        legend=dict(font=dict(size=9), bgcolor=LEGEND_BG, bordercolor="#e5e7eb", itemsizing="constant"),
    )
    return fig


def plot_violin_confidence(detections_df: pd.DataFrame) -> go.Figure:
    if detections_df.empty:
        return go.Figure()
    color_map = class_color_map()
    fig = go.Figure()
    for class_name, group in sorted(detections_df.groupby("class_name")):
        _c = color_map.get(class_name, "#888888")
        _h = _c.lstrip("#")
        _rgba = f"rgba({int(_h[0:2],16)},{int(_h[2:4],16)},{int(_h[4:6],16)},0.30)"
        fig.add_trace(go.Violin(
            y=group["confidence"], name=class_name,
            fillcolor=_rgba,
            line_color=_c,
            box_visible=True, meanline_visible=True, points="outliers",
            hovertemplate=f"<b>{class_name}</b><br>Conf: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Confidence Score Distribution per Class", font=dict(size=15, color=FONT_COLOR)),
        yaxis=dict(title="Confidence Score", range=[0, 1], gridcolor=GRID),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        height=420, showlegend=False, margin=dict(b=80),
    )
    return fig


# ── Inference / Video Charts ──────────────────────────────────────────────────

def plot_detections_per_frame(stats: dict) -> go.Figure:
    dpf = stats.get("detections_per_frame", [])
    if not dpf:
        return go.Figure()
    fig = go.Figure(go.Scatter(
        y=dpf, x=list(range(len(dpf))), mode="lines",
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
        line=dict(color="#2563eb", width=1.5),
        hovertemplate="Frame %{x}<br>Detections: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Detections per Frame", font=dict(size=15, color=FONT_COLOR)),
        xaxis=dict(title="Frame Index", gridcolor=GRID),
        yaxis=dict(title="Detections", gridcolor=GRID),
        height=260,
    )
    return fig


def plot_detection_donut(detections: list) -> go.Figure:
    if not detections:
        return go.Figure()
    from collections import Counter
    counts = Counter(d["class_name"] for d in detections)
    class_names = list(counts.keys())
    values = [counts[c] for c in class_names]
    color_map = class_color_map()
    colors = [color_map.get(c, "#888888") for c in class_names]
    fig = go.Figure(go.Pie(
        labels=class_names, values=values, hole=0.5,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Detected Classes", font=dict(size=14, color=FONT_COLOR)),
        height=280, showlegend=False, margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig

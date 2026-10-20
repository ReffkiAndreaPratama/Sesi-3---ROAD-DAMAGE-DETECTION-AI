"""
Modul visualisasi untuk RoadDamage AI — Bengkulu.

Berisi: penggambaran bounding box pada citra, penyusunan tampilan
sebelum/sesudah, dan grafik distribusi kelas (matplotlib & plotly).
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.utils import DetectionBox

# Palet warna stabil untuk kelas
_BOX_COLORS = [
    (0, 0, 255),      # merah
    (0, 255, 255),    # kuning
    (255, 0, 0),      # biru
    (0, 255, 0),      # hijau
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (128, 0, 255),    # ungu
    (0, 128, 255),    # oranye
    (255, 128, 0),    # biru muda
]


def color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Mengembalikan warna BGR stabil untuk nama kelas."""
    h = abs(hash(class_name)) % len(_BOX_COLORS)
    return _BOX_COLORS[h]


def annotate_image_bgr(
    image_bgr: np.ndarray,
    detections: Sequence[DetectionBox],
    line_thickness: int = 2,
    font_scale: float = 0.5,
    show_confidence: bool = True,
) -> np.ndarray:
    """
    Menggambar bounding box + label pada citra BGR.

    Mengembalikan salinan citra beranotasi. Tidak mengubah input.
    """
    import cv2

    annotated = image_bgr.copy()
    for d in detections:
        color = color_for_class(d.class_name)
        x1, y1, x2, y2 = int(round(d.x1)), int(round(d.y1)), int(round(d.x2)), int(round(d.y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        label = d.class_name
        if show_confidence:
            label = f"{d.class_name} {d.confidence:.2f}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        # Latar label
        y_label = max(y1, th + 4)
        cv2.rectangle(
            annotated,
            (x1, y_label - th - baseline - 2),
            (x1 + tw + 2, y_label + 2),
            color,
            -1,
        )
        cv2.putText(
            annotated, label, (x1 + 1, y_label - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return annotated


def annotate_image_rgb(
    image_rgb: np.ndarray,
    detections: Sequence[DetectionBox],
    **kwargs,
) -> np.ndarray:
    """Versi RGB dari annotate_image_bgr (mengkonversi masuk-keluar)."""
    bgr = image_rgb[:, :, ::-1].copy()
    annotated_bgr = annotate_image_bgr(bgr, detections, **kwargs)
    return annotated_bgr[:, :, ::-1].copy()


def make_before_after(
    original_rgb: np.ndarray,
    detections: Sequence[DetectionBox],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mengembalikan (asli_rgb, anotasi_rgb) untuk tampilan side-by-side."""
    annotated = annotate_image_rgb(original_rgb, detections)
    return original_rgb, annotated


def class_distribution_plotly(
    distribution: Dict[str, int],
    title: str = "Distribusi Kelas Kerusakan",
):
    """Membuat diagram batang Plotly untuk distribusi kelas."""
    import plotly.graph_objects as go

    classes = list(distribution.keys())
    counts = list(distribution.values())

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=counts,
                marker_color="indianred",
                text=counts,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Kelas Kerusakan",
        yaxis_title="Jumlah Deteksi",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def confidence_histogram_plotly(
    detections: Sequence[DetectionBox],
    title: str = "Histogram Confidence",
):
    """Membuat histogram confidence dari deteksi."""
    import plotly.graph_objects as go

    confs = [d.confidence for d in detections]
    fig = go.Figure(data=[go.Histogram(x=confs, nbinsx=20, marker_color="steelblue")])
    fig.update_layout(
        title=title,
        xaxis_title="Confidence",
        yaxis_title="Jumlah",
        template="plotly_white",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def confusion_matrix_plotly(
    matrix: "np.ndarray",
    class_names: Sequence[str],
    title: str = "Confusion Matrix",
):
    """Membuat heatmap Plotly untuk confusion matrix."""
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=np.array(matrix),
            x=list(class_names) + ["background"] if matrix.shape[1] == len(class_names) + 1 else list(class_names),
            y=list(class_names),
            colorscale="Blues",
            text=np.array(matrix),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Prediksi",
        yaxis_title="Ground Truth",
        height=520,
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=80),
    )
    return fig


def results_to_bar_chart(
    data: Dict[str, float],
    title: str = "Metrik Evaluasi",
    yrange: Tuple[float, float] = (0.0, 1.0),
):
    """Membuat diagram batang Plotly untuk metrik (precision, recall, mAP)."""
    import plotly.graph_objects as go

    names = list(data.keys())
    values = [float(v) for v in data.values()]

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=values,
                marker_color="seagreen",
                text=[f"{v:.3f}" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Metrik",
        yaxis_title="Nilai",
        yaxis=dict(range=list(yrange)),
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def image_to_png_bytes(image_rgb: np.ndarray) -> bytes:
    """Mengkonversi array RGB menjadi bytes PNG (untuk Streamlit download)."""
    from PIL import Image

    img = Image.fromarray(image_rgb.astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

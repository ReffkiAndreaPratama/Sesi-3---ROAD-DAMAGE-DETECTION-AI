"""
RoadDamage AI — Bengkulu
========================

Paket sumber untuk aplikasi deteksi kerusakan jalan berbasis YOLO.

Modul:
    - model_loader    : pemuatan bobot YOLO & deteksi perangkat.
    - image_detector   : inferensi citra tunggal.
    - batch_processor  : pemrosesan banyak citra sekaligus.
    - video_detector   : inferensi frame video.
    - dataset_utils    : konversi & validasi dataset kerusakan jalan.
    - evaluation       : evaluasi model (precision, recall, mAP).
    - visualization     : gambar bounding box & grafik.
    - utils            : utilitas umum (ekspor, validasi, logging).
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "model_loader",
    "image_detector",
    "video_detector",
    "batch_processor",
    "dataset_utils",
    "evaluation",
    "visualization",
    "utils",
]

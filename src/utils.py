"""
Utilitas umum untuk RoadDamage AI — Bengkulu.

Berisi: validasi citra, validasi parameter, ekspor hasil ke CSV/JSON,
konfigurasi logging, dan helper tipe data deteksi.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import settings


# ============================================================
# Logging
# ============================================================
def get_logger(name: str = "roaddamage") -> logging.Logger:
    """Mengembalikan logger yang dikonfigurasi (singleton-like)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


LOGGER = get_logger()


# ============================================================
# Dataclass hasil deteksi
# ============================================================
@dataclass
class DetectionBox:
    """Satu bounding box hasil deteksi."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_label_dict(self, img_w: int, img_h: int) -> Dict[str, Any]:
        """Format YOLO (normalisasi cx, cy, w, h)."""
        cx = (self.x1 + self.x2) / 2.0 / img_w
        cy = (self.y1 + self.y2) / 2.0 / img_h
        w = (self.x2 - self.x1) / img_w
        h = (self.y2 - self.y1) / img_h
        return {
            "class_id": self.class_id,
            "cx": round(cx, 6),
            "cy": round(cy, 6),
            "w": round(w, 6),
            "h": round(h, 6),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class DetectionResult:
    """Hasil deteksi untuk satu citra."""

    source: str
    image_width: int
    image_height: int
    detections: List[DetectionBox] = field(default_factory=list)
    inference_time: float = 0.0
    error: Optional[str] = None
    annotated_path: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def class_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for d in self.detections:
            dist[d.class_name] = dist.get(d.class_name, 0) + 1
        return dist

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "num_detections": self.num_detections,
            "inference_time": round(self.inference_time, 4),
            "error": self.error,
            "detections": [d.to_dict() for d in self.detections],
            "class_distribution": self.class_distribution(),
        }


# ============================================================
# Validasi citra
# ============================================================
def validate_image_file(path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Memvalidasi sebuah file citra.

    Mengembalikan (valid, pesan). Memeriksa ekstensi, ukuran, dan
    keterbacaan data citra.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in settings.SUPPORTED_IMAGE_EXTS:
        return False, (
            f"Ekstensi '{ext}' tidak didukung. "
            f"Gunakan salah satu: {', '.join(settings.SUPPORTED_IMAGE_EXTS)}"
        )

    if not path.exists():
        return False, f"File tidak ditemukan: {path}"

    size = path.stat().st_size
    if size > settings.MAX_IMAGE_BYTES:
        return False, (
            f"Ukuran file {size / (1024 * 1024):.1f} MB melebihi batas "
            f"{settings.MAX_IMAGE_BYTES / (1024 * 1024):.0f} MB."
        )
    if size == 0:
        return False, "File kosong (0 byte)."

    try:
        with Image.open(path) as img:
            img.verify()
    except UnidentifiedImageError:
        return False, "File bukan citra valid atau rusak."
    except Exception as e:
        return False, f"Gagal membaca citra: {e}"

    return True, "OK"


def load_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Memuat citra sebagai array BGR (format OpenCV). Mengembalikan None bila gagal."""
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        # Konversi RGB -> BGR untuk konsistensi OpenCV
        arr = arr[:, :, ::-1].copy()
        return arr
    except Exception as e:
        LOGGER.error("Gagal memuat citra %s: %s", path, e)
        return None


def load_image_rgb(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Memuat citra sebagai array RGB (format Pillow/Matplotlib)."""
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        LOGGER.error("Gagal memuat citra %s: %s", path, e)
        return None


def validate_video_file(path: Union[str, Path]) -> Tuple[bool, str]:
    """Memvalidasi file video berdasarkan ekstensi dan ukuran."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in settings.SUPPORTED_VIDEO_EXTS:
        return False, (
            f"Ekstensi video '{ext}' tidak didukung. "
            f"Gunakan salah satu: {', '.join(settings.SUPPORTED_VIDEO_EXTS)}"
        )

    if not path.exists():
        return False, f"File tidak ditemukan: {path}"

    size = path.stat().st_size
    if size > settings.MAX_VIDEO_BYTES:
        return False, (
            f"Ukuran video {size / (1024 * 1024):.1f} MB melebihi batas "
            f"{settings.MAX_VIDEO_BYTES / (1024 * 1024):.0f} MB."
        )
    return True, "OK"


# ============================================================
# Validasi parameter
# ============================================================
def validate_confidence(conf: float) -> Tuple[bool, str, float]:
    """
    Memvalidasi nilai confidence threshold.

    Mengembalikan (valid, pesan, nilai_clamped).
    """
    if conf is None:
        return True, "Menggunakan default.", settings.DEFAULT_CONFIDENCE
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return False, "Confidence harus berupa angka.", settings.DEFAULT_CONFIDENCE

    if conf < settings.MIN_CONFIDENCE:
        return False, (
            f"Confidence {conf} di bawah minimum {settings.MIN_CONFIDENCE}. "
            "Nilai dijepit ke minimum."
        ), settings.MIN_CONFIDENCE
    if conf > settings.MAX_CONFIDENCE:
        return False, (
            f"Confidence {conf} di atas maksimum {settings.MAX_CONFIDENCE}. "
            "Nilai dijepit ke maksimum."
        ), settings.MAX_CONFIDENCE
    return True, "OK", conf


def validate_image_size(size: int) -> Tuple[bool, str, int]:
    """Memvalidasi ukuran input gambar (harus bilangan positif)."""
    if size is None:
        return True, "Menggunakan default.", settings.DEFAULT_IMAGE_SIZE
    try:
        size = int(size)
    except (TypeError, ValueError):
        return False, "Ukuran harus berupa integer.", settings.DEFAULT_IMAGE_SIZE

    if size <= 0:
        return False, "Ukuran harus > 0.", settings.DEFAULT_IMAGE_SIZE
    if size > settings.MAX_IMAGE_SIZE:
        return False, (
            f"Ukuran {size} melebihi maksimum {settings.MAX_IMAGE_SIZE}."
        ), settings.MAX_IMAGE_SIZE
    return True, "OK", size


# ============================================================
# Ekspor hasil
# ============================================================
def export_results_csv(results: Sequence[DetectionResult], out_path: Union[str, Path]) -> Path:
    """
    Mengekspor daftar hasil deteksi ke CSV.

    Satu baris per deteksi individual.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for res in results:
        if res.is_error:
            rows.append({
                "source": res.source,
                "class_name": "",
                "confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "width": "", "height": "",
                "image_width": res.image_width,
                "image_height": res.image_height,
                "inference_time": res.inference_time,
                "status": res.error,
            })
            continue
        if res.num_detections == 0:
            rows.append({
                "source": res.source,
                "class_name": "(tidak ada deteksi)",
                "confidence": "",
                "x1": "", "y1": "", "x2": "", "y2": "",
                "width": "", "height": "",
                "image_width": res.image_width,
                "image_height": res.image_height,
                "inference_time": res.inference_time,
                "status": "OK",
            })
            continue
        for d in res.detections:
            rows.append({
                "source": res.source,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "x1": round(d.x1, 1), "y1": round(d.y1, 1),
                "x2": round(d.x2, 1), "y2": round(d.y2, 1),
                "width": round(d.width, 1), "height": round(d.height, 1),
                "image_width": res.image_width,
                "image_height": res.image_height,
                "inference_time": res.inference_time,
                "status": "OK",
            })

    fieldnames = [
        "source", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "width", "height",
        "image_width", "image_height", "inference_time", "status",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Hasil diekspor ke CSV: %s (%d baris)", out_path, len(rows))
    return out_path


def export_results_json(results: Sequence[DetectionResult], out_path: Union[str, Path]) -> Path:
    """Mengekspor daftar hasil deteksi ke JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "app": settings.APP_NAME,
        "exported_at": datetime.now().isoformat(),
        "total_images": len(results),
        "total_detections": sum(r.num_detections for r in results if not r.is_error),
        "results": [r.to_dict() for r in results],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    LOGGER.info("Hasil diekspor ke JSON: %s", out_path)
    return out_path


# ============================================================
# Helper umum
# ============================================================
def safe_filename(name: str, ext: str = "") -> str:
    """Membuat nama file aman dari string sembarang."""
    safe = "".join(c for c in name if c.isalnum() or c in ("_", "-", ".", " "))
    safe = safe.strip().replace(" ", "_")
    if ext and not safe.endswith(ext):
        safe += ext
    return safe or "output"


def timestamp_str() -> str:
    """Mengembalikan string waktu aman untuk nama file."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_size(num_bytes: int) -> str:
    """Memformat ukuran byte menjadi string mudah dibaca."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def is_cuda_available() -> bool:
    """Memeriksa ketersediaan CUDA secara aman."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_device(prefer_gpu: bool = False) -> str:
    """Mengembalikan 'cuda' / 'cuda:0' bila tersedia dan diminta, jika tidak 'cpu'."""
    if prefer_gpu and is_cuda_available():
        try:
            import torch
            return f"cuda:0" if torch.cuda.device_count() > 0 else "cpu"
        except Exception:
            pass
    return "cpu"


def get_supported_image_exts() -> List[str]:
    return sorted(settings.SUPPORTED_IMAGE_EXTS)


def get_supported_video_exts() -> List[str]:
    return sorted(settings.SUPPORTED_VIDEO_EXTS)

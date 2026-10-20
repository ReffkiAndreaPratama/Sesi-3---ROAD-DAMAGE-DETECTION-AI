"""
Konfigurasi terpusat untuk RoadDamage AI — Bengkulu.

Modul ini menyimpan metadata proyek, jalur default, definisi kelas
kerusakan jalan (berdasarkan Road Damage Dataset / RDD), serta parameter
deteksi dan batas validasi. Impor nilai dari sini agar tidak ada
konstanta yang tersebar di banyak file.
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# Metadata proyek
# ============================================================
APP_NAME = "RoadDamage AI — Bengkulu"
APP_TITLE = "RoadDamage AI — Bengkulu"
APP_SUBTITLE = "Evaluasi Model YOLO untuk Deteksi Kerusakan Jalan"
RESEARCH_TITLE = (
    "Evaluasi Kinerja Model YOLO untuk Deteksi Kerusakan Jalan "
    "Menggunakan Citra Digital di Kota Bengkulu"
)
APP_VERSION = "1.0.0"
APP_DESCRIPTION = (
    "Aplikasi deteksi kerusakan jalan berbasis YOLO dan Computer Vision. "
    "Hasil deteksi merupakan bantuan identifikasi, bukan pengganti inspeksi "
    "teknis jalan."
)

# ============================================================
# Struktur jalur (semua relatif terhadap root proyek)
# ============================================================
# Root proyek = folder induk dari folder "config" ini.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = PROJECT_ROOT / "config"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Subfolder data sesuai praktik YOLO
DATA_IMAGES_DIR = DATA_DIR / "images"
DATA_LABELS_DIR = DATA_DIR / "labels"
DATA_TRAIN_DIR = DATA_IMAGES_DIR / "train"
DATA_VAL_DIR = DATA_IMAGES_DIR / "val"
DATA_TEST_DIR = DATA_IMAGES_DIR / "test"
DATA_TRAIN_LABELS_DIR = DATA_LABELS_DIR / "train"
DATA_VAL_LABELS_DIR = DATA_LABELS_DIR / "val"
DATA_TEST_LABELS_DIR = DATA_LABELS_DIR / "test"

# Subfolder output
OUTPUTS_DETECTION_DIR = OUTPUTS_DIR / "detections"
OUTPUTS_EVAL_DIR = OUTPUTS_DIR / "evaluation"
OUTPUTS_VIDEO_DIR = OUTPUTS_DIR / "videos"
OUTPUTS_TRAIN_DIR = OUTPUTS_DIR / "training"

# ============================================================
# Definisi kelas kerusakan jalan (Road Damage Dataset / RDD)
# ============================================================
# Berdasarkan RDD2022 (sekilab/RoadDamageDetector). Id kelas string
# (D00..D40) dipetakan ke nama deskriptif. Catatan: pemetaan ini adalah
# panduan; sesuaikan dengan file data.yaml milik dataset yang Anda gunakan.
RDD_CLASSES = {
    "D00": "Retak memanjang (wheel mark)",
    "D01": "Retak memanjang (sambungan konstruksi)",
    "D10": "Retak melintang",
    "D11": "Sambungan konstruksi",
    "D20": "Retak buaya (alligator crack)",
    "D40": "Lepas butir / Lubang (raveling, pothole, bleeding)",
    "D44": "Lubang jalan (pothole)",
    "D50": "Garis putih pudar",
    "D51": "Tanda lajur rusak",
}

# Daftar nama kelas dalam urutan indeks (default YOLO class names).
# Digunakan saat membangun data.yaml untuk pelatihan.
DEFAULT_ROAD_DAMAGE_CLASSES = [
    "D00",
    "D01",
    "D10",
    "D11",
    "D20",
    "D40",
    "D44",
    "D50",
    "D51",
]

# Kelas yang lazim dianggap "kerusakan struktural" untuk ringkasan.
STRUCTURAL_DAMAGE_CLASSES = {"D00", "D01", "D10", "D11", "D20", "D40", "D44"}

# ============================================================
# Default deteksi
# ============================================================
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMAGE_SIZE = 640
MAX_IMAGE_SIZE = 4096
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# Batas ukuran unggahan
MAX_IMAGE_BYTES = 25 * 1024 * 1024      # 25 MB per citra
MAX_VIDEO_BYTES = 500 * 1024 * 1024    # 500 MB per video
MAX_BATCH_IMAGES = 50
MAX_VIDEO_FRAMES = 1500                 # batas aman pemrosesan frame
MAX_VIDEO_DURATION_SECONDS = 180        # batas durasi (detik)

# Ekstensi yang didukung
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# ============================================================
# Konfigurasi model
# ============================================================
# Nama bobot pralatih YOLO umum (deteksi objek) — HANYA contoh, bukan
# model kerusakan jalan. Digunakan untuk Mode A peringatan.
PRETRAINED_GENERAL_MODELS = {
    "yolov8n.pt": "YOLOv8 Nano (deteksi objek umum COCO)",
    "yolov8s.pt": "YOLOv8 Small (deteksi objek umum COCO)",
    "yolov8m.pt": "YOLOv8 Medium (deteksi objek umum COCO)",
    "yolov8l.pt": "YOLOv8 Large (deteksi objek umum COCO)",
    "yolov8x.pt": "YOLOv8 Extra Large (deteksi objek umum COCO)",
    "yolo11n.pt": "YOLO11 Nano (deteksi objek umum COCO)",
    "yolo11s.pt": "YOLO11 Small (deteksi objek umum COCO)",
    "yolo11m.pt": "YOLO11 Medium (deteksi objek umum COCO)",
}

# Jalur default bobot khusus kerusakan jalan (Mode B). Kosong sampai
# pengguna menempatkan bobot terlatih di folder models/.
DEFAULT_ROAD_DAMAGE_WEIGHTS = MODELS_DIR / "road_damage_best.pt"

# Konfigurasi pelatihan default
DEFAULT_TRAIN_EPOCHS = 100
DEFAULT_TRAIN_BATCH = 16
DEFAULT_TRAIN_IMG_SIZE = 640
DEFAULT_TRAIN_DEVICE = "cpu"   # CPU sebagai mode dasar; "0" untuk GPU pertama

# ============================================================
# Pengaturan runtime
# ============================================================
PREFER_GPU = False          # Default CPU; True jika CUDA terdeteksi & diizinkan
RANDOM_SEED = 42


def ensure_directories() -> None:
    """Membuat direktori utama proyek jika belum ada."""
    dirs = [
        DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SAMPLES_DIR,
        DATA_IMAGES_DIR, DATA_LABELS_DIR,
        DATA_TRAIN_DIR, DATA_VAL_DIR, DATA_TEST_DIR,
        DATA_TRAIN_LABELS_DIR, DATA_VAL_LABELS_DIR, DATA_TEST_LABELS_DIR,
        OUTPUTS_DETECTION_DIR, OUTPUTS_EVAL_DIR, OUTPUTS_VIDEO_DIR, OUTPUTS_TRAIN_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_output_path(subdir: str, filename: str) -> Path:
    """Mengembalikan jalur output lengkap dan memastikan folder ada."""
    target = OUTPUTS_DIR / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target / filename

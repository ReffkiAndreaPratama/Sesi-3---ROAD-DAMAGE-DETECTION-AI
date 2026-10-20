"""
Pemuat model YOLO untuk RoadDamage AI — Bengkulu.

Bertanggung jawab memuat bobot YOLO, memilih perangkat (CPU/GPU),
memvalidasi tugas deteksi, dan memeriksa kelas yang dikenali. Memberikan
peringatan jelas bila bobot adalah model deteksi objek umum (bukan
kerusakan jalan) atau bila bobot belum tersedia.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils import LOGGER, is_cuda_available, resolve_device
from config import settings

# ---------------------------------------------------------------------------
# Importasi YOLO di tingkat modul agar dapat di-patch oleh pengujian.
# Jika ultralytics belum terpasang, YOLO akan None dan loader akan memberikan
# pesan error yang jelas saat load() dipanggil.
# ---------------------------------------------------------------------------
try:
    from ultralytics import YOLO  # noqa: E402
except ImportError:
    YOLO = None  # type: ignore


# ============================================================
# Eksepsi khusus
# ============================================================
class ModelNotFoundError(FileNotFoundError):
    """Bobot model tidak ditemukan."""


class ModelLoadError(RuntimeError):
    """Gagal memuat model setelah file ditemukan."""


# ============================================================
# Dataclass status model
# ============================================================
@dataclass
class ModelStatus:
    """Status pemuatan model untuk ditampilkan di UI."""

    loaded: bool = False
    weights_path: str = ""
    model_type: str = ""          # "road_damage" | "general" | "unknown"
    device: str = "cpu"
    class_names: List[str] = field(default_factory=list)
    num_classes: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    backend: str = "ultralytics"

    @property
    def is_road_damage(self) -> bool:
        return self.model_type == "road_damage"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "weights_path": self.weights_path,
            "model_type": self.model_type,
            "device": self.device,
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "warnings": self.warnings,
            "errors": self.errors,
            "backend": self.backend,
        }


# ============================================================
# Kumpulan nama kelas COCO (model deteksi objek umum)
# ============================================================
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Kata kunci untuk mendeteksi apakah kelas adalah kelas kerusakan jalan
ROAD_DAMAGE_KEYWORDS = {
    "crack", "pothole", "rut", "damage", "raveling", "bleeding",
    "alligator", "longitudinal", "transverse", "d00", "d01", "d10",
    "d11", "d20", "d40", "d44", "d50", "d51", "kerusakan", "retak",
    "lubang", "jalan",
}


# ============================================================
# Pemuat model
# ============================================================
class ModelLoader:
    """Memuat dan mengelola model YOLO (Ultralytics)."""

    def __init__(self, prefer_gpu: bool = False) -> None:
        self.prefer_gpu = prefer_gpu
        self._model: Any = None
        self._status: ModelStatus = ModelStatus()
        self._ultralytics_ok = self._check_ultralytics()

    # ---------- Pemeriksaan ketersediaan backend ----------
    @staticmethod
    def _check_ultralytics() -> bool:
        return YOLO is not None

    @property
    def ultralytics_available(self) -> bool:
        return self._ultralytics_ok

    # ---------- Akses model ----------
    @property
    def model(self) -> Any:
        if self._model is None:
            raise ModelLoadError("Model belum dimuat. Panggil load() terlebih dahulu.")
        return self._model

    @property
    def status(self) -> ModelStatus:
        return self._status

    def is_loaded(self) -> bool:
        return self._model is not None and self._status.loaded

    # ---------- Pemuatan ----------
    def load(
        self,
        weights_path: Union[str, Path],
        prefer_gpu: Optional[bool] = None,
    ) -> ModelStatus:
        """
        Memuat bobot YOLO dari jalur.

        Bila prefer_gpu True dan CUDA tersedia, gunakan GPU; jika tidak, CPU.
        Mengembalikan ModelStatus dengan informasi diagnostik.
        """
        self._status = ModelStatus()
        self._model = None

        if prefer_gpu is not None:
            self.prefer_gpu = prefer_gpu

        if not self._ultralytics_ok or YOLO is None:
            self._status.errors.append(
                "Paket 'ultralytics' belum terpasang — tidak dapat memuat model. "
                "Jalankan 'pip install -r requirements.txt'."
            )
            LOGGER.error("Ultralytics tidak tersedia.")
            return self._status

        weights_path = Path(weights_path)
        self._status.weights_path = str(weights_path)

        # --- Bobot pralatih umum: biarkan Ultralytics mengunduh ---
        is_named = weights_path.name in settings.PRETRAINED_GENERAL_MODELS
        if not weights_path.exists() and not is_named:
            self._status.errors.append(
                f"File bobot tidak ditemukan: {weights_path}. "
                "Letakkan bobot terlatih di folder 'models/' atau pilih "
                "bobot pralatih. Lihat README bagian 'Menyiapkan Bobot Model'."
            )
            LOGGER.error("Bobot tidak ditemukan: %s", weights_path)
            return self._status

        try:
            LOGGER.info("Memuat bobot: %s", weights_path)
            self._model = YOLO(str(weights_path))
        except Exception as e:
            self._model = None
            self._status.errors.append(f"Gagal memuat model YOLO: {e}")
            LOGGER.exception("Gagal memuat model YOLO")
            return self._status

        # --- Pilih perangkat ---
        device = resolve_device(self.prefer_gpu)
        self._status.device = device
        try:
            self._model.to(device)
        except Exception:
            # Beberapa versi tidak punya .to(); inferensi akan tetap menerima device.
            pass

        # --- Ekstrak nama kelas & jenis model ---
        class_names = self._extract_class_names()
        self._status.class_names = class_names
        self._status.num_classes = len(class_names)

        self._status.model_type = self._classify_model_type(weights_path, class_names)
        self._status.loaded = True
        self._status.backend = "ultralytics"

        # --- Peringatan jika model umum ---
        if self._status.model_type == "general":
            self._status.warnings.append(
                "Model ini adalah model deteksi objek umum (COCO). Bobot ini "
                "TIDAK dilatih untuk mendeteksi kerusakan jalan. Hasil deteksi "
                "kemungkinan hanya mendeteksi kendaraan/orang, bukan retak atau "
                "lubang. Gunakan bobot khusus kerusakan jalan (Mode B) untuk "
                "deteksi yang relevan."
            )
            LOGGER.warning("Model umum dimuat — bukan model kerusakan jalan.")
        elif self._status.model_type == "unknown":
            self._status.warnings.append(
                "Jenis model tidak dapat dipastikan. Pastikan bobot benar-benar "
                "dilatih untuk kelas kerusakan jalan sebelum menyimpulkan hasil."
            )

        LOGGER.info(
            "Model dimuat: tipe=%s, kelas=%d, perangkat=%s",
            self._status.model_type, self._status.num_classes, device,
        )
        return self._status

    # ---------- Ekstraksi nama kelas ----------
    def _extract_class_names(self) -> List[str]:
        names: Any = None
        try:
            names = getattr(self._model, "names", None)
        except Exception:
            names = None

        if names is None:
            return []

        if isinstance(names, dict):
            # {0: 'D00', 1: 'D01', ...}
            return [str(names[i]) for i in sorted(names.keys())]
        if isinstance(names, (list, tuple)):
            return [str(n) for n in names]
        return []

    # ---------- Klasifikasi jenis model ----------
    @staticmethod
    def _classify_model_type(weights_path: Path, class_names: List[str]) -> str:
        name = weights_path.name

        # Nama bobot pralatih umum jelas bukan kerusakan jalan
        if name in settings.PRETRAINED_GENERAL_MODELS:
            return "general"

        # Apakah kelas terlihat seperti kelas kerusakan jalan?
        if class_names:
            lower_names = [c.lower() for c in class_names]
            road_hits = sum(
                1
                for n in lower_names
                if any(kw in n for kw in ROAD_DAMAGE_KEYWORDS)
            )
            coco_hits = sum(1 for n in lower_names if n in set(COCO_CLASSES))

            if road_hits >= max(1, len(class_names) // 2):
                return "road_damage"
            if coco_hits >= max(1, len(class_names) // 2):
                return "general"

        # Jika nama file mengandung petunjuk
        lower_name = name.lower()
        if any(kw in lower_name for kw in ("road", "damage", "rdd", "crack")):
            return "road_damage"

        return "unknown"


# ============================================================
# Helper publik
# ============================================================
def list_available_weights(models_dir: Union[str, Path] = None) -> List[str]:
    """Mendaftar file bobot (.pt/.pth) yang ada di folder models/."""
    if models_dir is None:
        models_dir = settings.MODELS_DIR
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    weights = []
    for ext in ("*.pt", "*.pth"):
        weights.extend(p.name for p in models_dir.glob(ext))
    return sorted(set(weights))


def get_pretrained_options() -> Dict[str, str]:
    """Mengembalikan pilihan bobot pralatih umum (Mode A)."""
    return dict(settings.PRETRAINED_GENERAL_MODELS)


def check_weights_exists(weights_path: Union[str, Path]) -> Tuple[bool, str]:
    """Memeriksa apakah file bobot ada (atau nama pralatih yang akan diunduh)."""
    weights_path = Path(weights_path)
    if weights_path.name in settings.PRETRAINED_GENERAL_MODELS:
        return True, f"Bobot pralatih '{weights_path.name}' (akan diunduh otomatis)."
    if weights_path.exists():
        return True, f"Bobot ditemukan: {weights_path}"
    return False, (
        f"Bobot tidak ditemukan: {weights_path}. Letakkan bobot terlatih di "
        "folder 'models/' atau pilih bobot pralatih. Lihat README."
    )

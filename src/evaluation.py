"""
Modul evaluasi model untuk RoadDamage AI — Bengkulu.

Menjalankan validasi YOLO pada dataset berlabel dan mengekstrak metrik
precision, recall, mAP@0.5, mAP@0.5:0.95, serta confusion matrix.

PENTING: metrik diambil dari hasil validasi YOLO yang sebenarnya. Jika
dataset evaluasi (ground truth) belum tersedia, fungsi mengembalikan
status "belum dilakukan" dan TIDAK mengarang nilai.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.dataset_utils import DatasetReport, validate_yolo_dataset
from src.model_loader import ModelLoader, ModelStatus
from src.utils import LOGGER
from config import settings


@dataclass
class EvaluationMetrics:
    """Hasil evaluasi model."""

    done: bool = False
    available: bool = False
    message: str = ""
    num_images: int = 0
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    precision: Optional[float] = None
    recall: Optional[float] = None
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    fitness: Optional[float] = None
    per_class: List[Dict[str, Any]] = field(default_factory=list)
    confusion_matrix: Optional[List[List[float]]] = None
    confusion_matrix_classes: List[str] = field(default_factory=list)
    inference_speed_ms: Optional[float] = None
    evaluation_time_seconds: float = 0.0
    data_yaml: str = ""
    device: str = "cpu"
    errors: List[str] = field(default_factory=list)

    @property
    def has_metrics(self) -> bool:
        return self.precision is not None or self.map50 is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "done": self.done,
            "available": self.available,
            "message": self.message,
            "num_images": self.num_images,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "precision": self.precision,
            "recall": self.recall,
            "map50": self.map50,
            "map50_95": self.map50_95,
            "fitness": self.fitness,
            "per_class": self.per_class,
            "inference_speed_ms": self.inference_speed_ms,
            "evaluation_time_seconds": round(self.evaluation_time_seconds, 3),
            "data_yaml": self.data_yaml,
            "device": self.device,
            "errors": self.errors,
        }


class ModelEvaluator:
    """Mengevaluasi model YOLO pada dataset berlabel."""

    def __init__(self, loader: ModelLoader) -> None:
        self.loader = loader

    @property
    def status(self) -> ModelStatus:
        return self.loader.status

    def evaluate(
        self,
        data_yaml: Union[str, Path],
        image_size: int = settings.DEFAULT_IMAGE_SIZE,
        batch: int = 1,
        device: Optional[str] = None,
        split: str = "test",
        save_dir: Optional[Union[str, Path]] = None,
    ) -> EvaluationMetrics:
        """
        Menjalankan validasi YOLO pada dataset berlabel.

        Parameter:
            data_yaml  : jalur ke data.yaml (ground truth harus ada).
            image_size : ukuran input evaluasi.
            batch      : ukuran batch evaluasi.
            device     : perangkat inferensi.
            split      : split dataset yang dievaluasi ('val'/'test').
            save_dir   : folder penyimpanan hasil evaluasi.
        """
        metrics = EvaluationMetrics()

        # --- Validasi model ---
        if not self.loader.is_loaded():
            metrics.errors.append(
                "Model belum dimuat. Muat bobot di halaman Pengaturan Model."
            )
            metrics.message = "Model belum dimuat."
            return metrics

        data_yaml = Path(data_yaml)
        metrics.data_yaml = str(data_yaml)
        if not data_yaml.exists():
            metrics.errors.append(
                f"File data.yaml tidak ditemukan: {data_yaml}. "
                "Siapkan dataset berlabel sebelum mengevaluasi. "
                "Lihat README bagian 'Dataset'."
            )
            metrics.message = "Dataset evaluasi belum tersedia."
            return metrics

        # --- Validasi dataset ada isinya ---
        report = validate_yolo_dataset(settings.DATA_DIR)
        split_images = report.images_per_split.get(split, 0)
        metrics.num_images = split_images
        if split_images == 0:
            metrics.errors.append(
                f"Tidak ada citra pada split '{split}'. Evaluasi membutuhkan "
                "ground truth (anotasi) yang valid pada split terpisah dari data latih."
            )
            metrics.message = (
                f"Split '{split}' kosong. Evaluasi belum dapat dilakukan."
            )
            return metrics

        # --- Jalankan validasi ---
        try:
            model = self.loader.model
            infer_device = device or self.status.device
            LOGGER.info("Memulai evaluasi: data=%s split=%s", data_yaml, split)
            t0 = time.perf_counter()

            val_args = dict(
                data=str(data_yaml),
                imgsz=image_size,
                batch=batch,
                device=infer_device,
                split=split,
                verbose=False,
                plots=False,
            )
            if save_dir is not None:
                val_args["project"] = str(Path(save_dir).parent)
                val_args["name"] = Path(save_dir).name

            results = model.val(**val_args)
            metrics.evaluation_time_seconds = time.perf_counter() - t0
        except Exception as e:
            LOGGER.exception("Evaluasi gagal")
            metrics.errors.append(f"Evaluasi gagal: {e}")
            metrics.message = "Evaluasi gagal dijalankan."
            return metrics

        # --- Ekstrak metrik ---
        self._extract_metrics(results, metrics)
        metrics.done = True
        metrics.available = metrics.has_metrics

        if not metrics.has_metrics:
            metrics.message = (
                "Evaluasi selesai tetapi metrik tidak dapat diekstrak. "
                "Periksa format dataset dan anotasi."
            )
        else:
            metrics.message = "Evaluasi berhasil dilakukan."

        LOGGER.info(
            "Evaluasi selesai: P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f",
            metrics.precision or 0, metrics.recall or 0,
            metrics.map50 or 0, metrics.map50_95 or 0,
        )
        return metrics

    # --------------------------------------------------------
    def _extract_metrics(self, results: Any, metrics: EvaluationMetrics) -> None:
        """Mengekstrak metrik dari objek hasil YOLO val()."""
        if results is None:
            return

        # Nama kelas dari model
        try:
            metrics.class_names = list(self.status.class_names)
            metrics.num_classes = len(metrics.class_names)
        except Exception:
            pass

        # Box metrics (P, R, mAP) — bisa dict atau atribut
        try:
            box = getattr(results, "box", None)
        except Exception:
            box = None

        if box is not None:
            try:
                metrics.precision = float(_safe_get(box, "mp")) if _safe_get(box, "mp") is not None else None
            except (TypeError, ValueError):
                metrics.precision = None
            try:
                metrics.recall = float(_safe_get(box, "mr")) if _safe_get(box, "mr") is not None else None
            except (TypeError, ValueError):
                metrics.recall = None
            try:
                metrics.map50 = float(_safe_get(box, "map50")) if _safe_get(box, "map50") is not None else None
            except (TypeError, ValueError):
                metrics.map50 = None
            try:
                metrics.map50_95 = float(_safe_get(box, "map")) if _safe_get(box, "map") is not None else None
            except (TypeError, ValueError):
                metrics.map50_95 = None

            # Per-class metrics
            self._extract_per_class(box, metrics)

        # Fitness
        try:
            metrics.fitness = float(getattr(results, "fitness", None))
            if metrics.fitness != metrics.fitness:  # NaN
                metrics.fitness = None
        except (TypeError, ValueError):
            metrics.fitness = None

        # Kecepatan inferensi
        try:
            speed = getattr(results, "speed", None)
            if speed is not None:
                # speed = dict(inference=..., preprocess=..., postprocess=...)
                infer_ms = _safe_get(speed, "inference")
                if infer_ms is not None:
                    metrics.inference_speed_ms = float(infer_ms)
        except Exception:
            pass

        # Confusion matrix
        try:
            cm = getattr(results, "confusion_matrix", None)
            if cm is not None:
                matrix = getattr(cm, "matrix", None)
                if matrix is not None:
                    import numpy as np
                    arr = np.array(matrix).tolist()
                    metrics.confusion_matrix = arr
                    # Label kelas (tambah 'background' jika ada kolom ekstra)
                    names = list(metrics.class_names)
                    if arr and len(arr[0]) == len(names) + 1:
                        names = names + ["background"]
                    metrics.confusion_matrix_classes = names
        except Exception as e:
            LOGGER.debug("Confusion matrix tidak diekstrak: %s", e)

    def _extract_per_class(self, box: Any, metrics: EvaluationMetrics) -> None:
        """Mengekstrak metrik per kelas dari results.box."""
        try:
            import numpy as np
            p_arr = getattr(box, "p", None)     # per-class precision (array)
            r_arr = getattr(box, "r", None)     # per-class recall
            ap50_arr = getattr(box, "ap50", None)
            ap_arr = getattr(box, "ap", None)
            names = metrics.class_names

            n = len(names) if names else 0
            if n == 0:
                # Coba dapatkan panjang dari array
                for arr in (p_arr, r_arr, ap50_arr, ap_arr):
                    if arr is not None and hasattr(arr, "__len__"):
                        n = len(arr)
                        names = [f"class_{i}" for i in range(n)]
                        break
            if n == 0:
                return

            for i in range(n):
                cls_name = names[i] if i < len(names) else f"class_{i}"
                row = {
                    "class": cls_name,
                    "precision": _safe_float_idx(p_arr, i),
                    "recall": _safe_float_idx(r_arr, i),
                    "ap50": _safe_float_idx(ap50_arr, i),
                    "ap50_95": _safe_float_idx(ap_arr, i),
                }
                metrics.per_class.append(row)
        except Exception as e:
            LOGGER.debug("Per-class extraction gagal: %s", e)


# ============================================================
# Helper
# ============================================================
def _safe_get(obj: Any, key: str) -> Any:
    """Mengakses atribut atau kunci dict dengan aman."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _safe_float_idx(arr: Any, idx: int) -> Optional[float]:
    """Mengambil float dari array pada indeks, aman terhadap error."""
    if arr is None:
        return None
    try:
        import numpy as np
        if isinstance(arr, np.ndarray):
            arr = arr.tolist()
        v = arr[idx]
        v = float(v)
        return v if v == v else None  # filter NaN
    except (IndexError, TypeError, ValueError):
        return None


# ============================================================
# Fungsi publik
# ============================================================
def get_evaluation_status_text(metrics: EvaluationMetrics) -> str:
    """Mengembalikan teks status evaluasi yang jujur untuk UI."""
    if not metrics.done:
        return (
            "Evaluasi belum dilakukan. Dataset berlabel (ground truth) "
            "diperlukan untuk menghitung precision, recall, dan mAP. "
            "Nilai metrik tidak akan ditampilkan sebelum evaluasi nyata dijalankan."
        )
    if metrics.errors and not metrics.has_metrics:
        return "Evaluasi gagal: " + "; ".join(metrics.errors)
    if metrics.has_metrics:
        return (
            f"Evaluasi berhasil pada {metrics.num_images} citra. "
            f"Precision={metrics.precision:.4f}, Recall={metrics.recall:.4f}, "
            f"mAP@0.5={metrics.map50:.4f}, mAP@0.5:0.95={metrics.map50_95:.4f}."
        )
    return "Evaluasi selesai tetapi metrik tidak tersedia."

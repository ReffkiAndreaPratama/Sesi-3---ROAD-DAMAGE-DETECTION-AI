"""
Pemroses batch citra untuk RoadDamage AI — Bengkulu.

Memproses banyak citra sekaligus, melacak status pemrosesan,
menyusun tabel ringkasan, dan mengekspor rekap CSV/JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from src.image_detector import ImageDetector
from src.utils import (
    DetectionResult,
    LOGGER,
    export_results_csv,
    export_results_json,
    timestamp_str,
)
from config import settings


@dataclass
class BatchProgress:
    """Status pemrosesan batch."""

    total: int = 0
    done: int = 0
    current_file: str = ""
    errors: int = 0

    @property
    def percent(self) -> float:
        return 100.0 * self.done / self.total if self.total else 0.0


class BatchProcessor:
    """Memproses banyak citra sekaligus."""

    def __init__(self, detector: ImageDetector) -> None:
        self.detector = detector
        self.results: List[DetectionResult] = []
        self.progress = BatchProgress()

    def process(
        self,
        image_paths: Sequence[Union[str, Path]],
        confidence: float = settings.DEFAULT_CONFIDENCE,
        iou: float = settings.DEFAULT_IOU,
        image_size: int = settings.DEFAULT_IMAGE_SIZE,
        save_annotated: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
        callback=None,
    ) -> Tuple[List[DetectionResult], BatchProgress]:
        """
        Memproses daftar citra.

        callback: callable(progress: BatchProgress) opsional untuk UI.
        Mengembalikan (hasil, progress).
        """
        self.results = []
        self.progress = BatchProgress(total=min(len(image_paths), settings.MAX_BATCH_IMAGES))

        if not image_paths:
            return self.results, self.progress

        paths = list(image_paths)[: settings.MAX_BATCH_IMAGES]

        for p in paths:
            p = Path(p)
            self.progress.current_file = p.name

            if self.progress.total > settings.MAX_BATCH_IMAGES and \
                    paths is image_paths and len(image_paths) > settings.MAX_BATCH_IMAGES:
                pass

            result = self.detector.detect(
                image_path=p,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
                save_annotated=save_annotated,
                output_dir=output_dir,
            )
            self.results.append(result)
            self.progress.done += 1
            if result.is_error:
                self.progress.errors += 1

            if callback is not None:
                try:
                    callback(self.progress)
                except Exception:
                    pass

        LOGGER.info(
            "Batch selesai: %d/%d citra, %d error",
            self.progress.done, self.progress.total, self.progress.errors,
        )
        return self.results, self.progress

    def export_csv(self, out_path: Optional[Union[str, Path]] = None) -> Path:
        """Mengekspor rekap hasil batch ke CSV."""
        if out_path is None:
            out_path = settings.OUTPUTS_DIR / f"batch_results_{timestamp_str()}.csv"
        return export_results_csv(self.results, out_path)

    def export_json(self, out_path: Optional[Union[str, Path]] = None) -> Path:
        """Mengekspor rekap hasil batch ke JSON."""
        if out_path is None:
            out_path = settings.OUTPUTS_DIR / f"batch_results_{timestamp_str()}.json"
        return export_results_json(self.results, out_path)

    def summary_table(self) -> List[Dict[str, object]]:
        """Mengembalikan tabel ringkasan per citra (untuk UI Streamlit)."""
        rows: List[Dict[str, object]] = []
        for r in self.results:
            rows.append({
                "File": r.source,
                "Jumlah Deteksi": 0 if r.is_error else r.num_detections,
                "Kelas Terdeteksi": ", ".join(r.class_distribution().keys()) if not r.is_error else "",
                "Status": r.error if r.is_error else "OK",
                "Waktu Inferensi (s)": round(r.inference_time, 3),
            })
        return rows

    def aggregate_distribution(self) -> Dict[str, int]:
        """Menggabungkan distribusi kelas dari semua citra sukses."""
        agg: Dict[str, int] = {}
        for r in self.results:
            if r.is_error:
                continue
            for cls, cnt in r.class_distribution().items():
                agg[cls] = agg.get(cls, 0) + cnt
        return agg


def validate_batch_files(paths: Sequence[Union[str, Path]]) -> Tuple[List[Path], List[str]]:
    """
    Memvalidasi daftar file batch.

    Mengembalikan (valid_paths, error_messages).
    """
    from src.utils import validate_image_file

    valid: List[Path] = []
    errors: List[str] = []
    for p in paths:
        p = Path(p)
        ok, msg = validate_image_file(p)
        if ok:
            valid.append(p)
        else:
            errors.append(f"{p.name}: {msg}")
    return valid, errors

#!/usr/bin/env python
"""
Skrip prediksi deteksi kerusakan jalan pada citra.

Penggunaan (dari root proyek):
    # Satu citra
    python -m scripts.predict --weights models/road_damage_best.pt \\
        --input path/to/road.jpg --conf 0.25

    # Folder citra (batch)
    python -m scripts.predict --weights models/road_damage_best.pt \\
        --input path/to/folder --conf 0.25 --save-csv

Menyimpan citra beranotasi dan rekap CSV/JSON di folder outputs/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_loader import ModelLoader
from src.image_detector import ImageDetector
from src.batch_processor import BatchProcessor, validate_batch_files
from src.utils import LOGGER, export_results_csv, export_results_json, timestamp_str
from config import settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prediksi deteksi kerusakan jalan.")
    p.add_argument("--weights", type=str, default=str(settings.DEFAULT_ROAD_DAMAGE_WEIGHTS),
                   help="Jalur bobot model (.pt).")
    p.add_argument("--input", type=str, required=True,
                   help="Satu citra atau folder berisi citra.")
    p.add_argument("--conf", type=float, default=settings.DEFAULT_CONFIDENCE)
    p.add_argument("--iou", type=float, default=settings.DEFAULT_IOU)
    p.add_argument("--imgsz", type=int, default=settings.DEFAULT_IMAGE_SIZE)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save-csv", action="store_true", default=True)
    p.add_argument("--save-json", action="store_true", default=True)
    p.add_argument("--no-save-annotated", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    # --- Muat model ---
    loader = ModelLoader()
    status = loader.load(args.weights)
    if not status.loaded:
        LOGGER.error("Model tidak dapat dimuat: %s", "; ".join(status.errors))
        return 2

    detector = ImageDetector(loader)

    # --- Tentukan apakah input file atau folder ---
    if input_path.is_file():
        result = detector.detect(
            image_path=input_path,
            confidence=args.conf,
            iou=args.iou,
            image_size=args.imgsz,
            device=args.device,
            save_annotated=not args.no_save_annotated,
        )
        results = [result]
    elif input_path.is_dir():
        all_images = []
        for ext in settings.SUPPORTED_IMAGE_EXTS:
            all_images.extend(sorted(input_path.glob(f"*{ext}")))
        if not all_images:
            LOGGER.error("Tidak ada citra yang ditemukan di %s", input_path)
            return 1
        valid, errs = validate_batch_files(all_images)
        if errs:
            for e in errs:
                LOGGER.warning("Lewati: %s", e)
        if not valid:
            LOGGER.error("Tidak ada citra valid untuk diproses.")
            return 1
        batch = BatchProcessor(detector)
        results, progress = batch.process(
            valid,
            confidence=args.conf,
            iou=args.iou,
            image_size=args.imgsz,
            save_annotated=not args.no_save_annotated,
        )
        LOGGER.info("Batch: %d/%d selesai, %d error",
                    progress.done, progress.total, progress.errors)
    else:
        LOGGER.error("Input tidak ditemukan: %s", input_path)
        return 1

    # --- Ringkasan ---
    total_det = sum(r.num_detections for r in results if not r.is_error)
    print("\n" + "=" * 60)
    print("HASIL PREDIKSI ROAD DAMAGE AI — BENGKULU")
    print("=" * 60)
    for r in results:
        if r.is_error:
            print(f"  [ERROR] {r.source}: {r.error}")
        else:
            print(f"  {r.source}: {r.num_detections} deteksi "
                  f"({', '.join(f'{k}={v}' for k, v in r.class_distribution().items()) or 'tidak ada deteksi'})")
    print(f"Total deteksi: {total_det}")
    print("=" * 60)

    # --- Ekspor ---
    if args.save_csv:
        out = settings.OUTPUTS_DIR / f"predict_{timestamp_str()}.csv"
        export_results_csv(results, out)
        print(f"CSV: {out}")
    if args.save_json:
        out = settings.OUTPUTS_DIR / f"predict_{timestamp_str()}.json"
        export_results_json(results, out)
        print(f"JSON: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

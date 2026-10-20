#!/usr/bin/env python
"""
Skrip pelatihan model YOLO untuk deteksi kerusakan jalan.

Penggunaan (dari root proyek):
    python -m scripts.train --data data/data.yaml --epochs 100 --batch 16 \\
        --imgsz 640 --device cpu --weights yolov8s.pt --name road_damage

Membutuhkan dataset format YOLO (lihat README bagian 'Dataset').
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Memastikan root proyek ada di sys.path saat dijalankan langsung
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_loader import ModelLoader
from src.dataset_utils import validate_yolo_dataset, write_data_yaml
from src.utils import LOGGER, is_cuda_available
from config import settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Latih model YOLO untuk kerusakan jalan.")
    p.add_argument("--data", type=str, default=None,
                   help="Jalur ke data.yaml (default: data/data.yaml).")
    p.add_argument("--weights", type=str, default="yolov8s.pt",
                   help="Bobot awal (default: yolov8s.pt).")
    p.add_argument("--epochs", type=int, default=settings.DEFAULT_TRAIN_EPOCHS)
    p.add_argument("--batch", type=int, default=settings.DEFAULT_TRAIN_BATCH)
    p.add_argument("--imgsz", type=int, default=settings.DEFAULT_TRAIN_IMG_SIZE)
    p.add_argument("--device", type=str, default=settings.DEFAULT_TRAIN_DEVICE,
                   help="'cpu' atau '0' untuk GPU pertama.")
    p.add_argument("--name", type=str, default="road_damage",
                   help="Nama eksperimen (folder output).")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--resume", action="store_true", help="Lanjutkan pelatihan terakhir.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --- Validasi backend ---
    loader = ModelLoader()
    if not loader.ultralytics_available:
        LOGGER.error("ultralytics belum terpasang. Jalankan: pip install -r requirements.txt")
        return 1

    # --- Tentukan data.yaml ---
    data_yaml = Path(args.data) if args.data else settings.DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        LOGGER.info("data.yaml belum ada. Mencoba membuat otomatis...")
        # Validasi dataset ada
        report = validate_yolo_dataset(settings.DATA_DIR)
        if not report.found or report.total_images == 0:
            LOGGER.error(
                "Dataset belum tersedia di '%s'. Siapkan dataset format YOLO "
                "(images/<split> + labels/<split>) sebelum melatih. Lihat README.",
                settings.DATA_DIR,
            )
            return 1
        data_yaml = write_data_yaml(settings.DATA_DIR)
    LOGGER.info("Menggunakan data.yaml: %s", data_yaml)

    # --- Perangkat ---
    device = args.device
    if device not in ("cpu",) and not is_cuda_available():
        LOGGER.warning("CUDA tidak tersedia; beralih ke CPU.")
        device = "cpu"

    # --- Jalankan pelatihan ---
    try:
        from ultralytics import YOLO
        LOGGER.info("Memulai pelatihan: epochs=%d batch=%d imgsz=%d device=%s",
                    args.epochs, args.batch, args.imgsz, device)
        model = YOLO(args.weights)
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            name=args.name,
            workers=args.workers,
            resume=args.resume,
            project=str(settings.OUTPUTS_TRAIN_DIR),
            exist_ok=True,
        )
        LOGGER.info("Pelatihan selesai. Bobot tersimpan di: %s",
                     settings.OUTPUTS_TRAIN_DIR)
        LOGGER.info("Salin file best.pt ke models/road_damage_best.pt untuk dipakai aplikasi.")
        return 0
    except Exception as e:
        LOGGER.exception("Pelatihan gagal: %s", e)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

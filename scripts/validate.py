#!/usr/bin/env python
"""
Skrip evaluasi model YOLO untuk deteksi kerusakan jalan.

Penggunaan (dari root proyek):
    python -m scripts.validate --weights models/road_damage_best.pt \\
        --data data/data.yaml --split test --imgsz 640

Membutuhkan dataset berlabel (ground truth) yang terpisah dari data latih.
Nilai metrik diambil dari hasil validasi yang sebenarnya; tidak ada nilai
yang diarang.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_loader import ModelLoader
from src.evaluation import ModelEvaluator, get_evaluation_status_text
from src.dataset_utils import validate_yolo_dataset, write_data_yaml
from src.utils import LOGGER
from config import settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluasi model YOLO kerusakan jalan.")
    p.add_argument("--weights", type=str, default=str(settings.DEFAULT_ROAD_DAMAGE_WEIGHTS),
                   help="Jalur bobot model (.pt).")
    p.add_argument("--data", type=str, default=None,
                   help="Jalur ke data.yaml (default: data/data.yaml).")
    p.add_argument("--split", type=str, default="test",
                   choices=["val", "test"], help="Split yang dievaluasi.")
    p.add_argument("--imgsz", type=int, default=settings.DEFAULT_IMAGE_SIZE)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", type=str, default=None,
                   help="'cpu' atau '0' untuk GPU.")
    p.add_argument("--save-json", action="store_true",
                   help="Simpan hasil evaluasi ke JSON.")
    p.add_argument("--out-dir", type=str, default=str(settings.OUTPUTS_EVAL_DIR))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # --- Tentukan data.yaml ---
    data_yaml = Path(args.data) if args.data else settings.DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        report = validate_yolo_dataset(settings.DATA_DIR)
        if not report.found or report.total_images == 0:
            LOGGER.error(
                "Dataset evaluasi belum tersedia di '%s'. Evaluasi membutuhkan "
                "ground truth pada split '%s'. Lihat README.",
                settings.DATA_DIR, args.split,
            )
            return 1
        data_yaml = write_data_yaml(settings.DATA_DIR)

    # --- Muat model ---
    loader = ModelLoader()
    status = loader.load(args.weights)
    if not status.loaded:
        LOGGER.error("Model tidak dapat dimuat: %s", "; ".join(status.errors))
        return 2

    # --- Evaluasi ---
    evaluator = ModelEvaluator(loader)
    save_dir = Path(args.out_dir) / "run"
    metrics = evaluator.evaluate(
        data_yaml=data_yaml,
        image_size=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        save_dir=save_dir,
    )

    print("\n" + "=" * 60)
    print("HASIL EVALUASI ROAD DAMAGE AI — BENGKULU")
    print("=" * 60)
    print(get_evaluation_status_text(metrics))
    if metrics.has_metrics:
        print(f"Jumlah citra    : {metrics.num_images}")
        print(f"Precision       : {metrics.precision:.4f}" if metrics.precision is not None else "Precision       : N/A")
        print(f"Recall          : {metrics.recall:.4f}" if metrics.recall is not None else "Recall          : N/A")
        print(f"mAP@0.5         : {metrics.map50:.4f}" if metrics.map50 is not None else "mAP@0.5         : N/A")
        print(f"mAP@0.5:0.95    : {metrics.map50_95:.4f}" if metrics.map50_95 is not None else "mAP@0.5:0.95    : N/A")
        if metrics.inference_speed_ms is not None:
            print(f"Inferensi (ms)  : {metrics.inference_speed_ms:.2f}")
        if metrics.per_class:
            print("\nMetrik per kelas:")
            print(f"  {'Kelas':<22}{'P':>8}{'R':>8}{'mAP50':>8}{'mAP50-95':>10}")
            for row in metrics.per_class:
                print(f"  {str(row['class']):<22}"
                      f"{_fmt(row['precision'])}{'':<0}"
                      f"{_fmt(row['recall'])}{'':<0}"
                      f"{_fmt(row['ap50'])}{'':<0}"
                      f"{_fmt(row['ap50_95'])}")
    print("=" * 60)

    if metrics.errors:
        for e in metrics.errors:
            print(f"  ! {e}")

    if args.save_json:
        out = Path(args.out_dir) / f"eval_{metrics.num_images}img.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\nHasil disimpan: {out}")

    return 0 if metrics.has_metrics else 1


def _fmt(v):
    if v is None:
        return f"{'N/A':>8}"
    return f"{v:>8.4f}"


if __name__ == "__main__":
    raise SystemExit(main())

"""
Detektor citra tunggal untuk RoadDamage AI — Bengkulu.

Membungkus inferensi YOLO untuk satu citra, mengonversi output mentah
menjadi DetectionResult yang terstruktur, dan menyimpan citra beranotasi.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from src.model_loader import ModelLoader, ModelLoadError, ModelStatus
from src.utils import (
    DetectionBox,
    DetectionResult,
    LOGGER,
    load_image,
    load_image_rgb,
    safe_filename,
    timestamp_str,
    validate_confidence,
    validate_image_file,
    validate_image_size,
)
from src.visualization import annotate_image_bgr
from config import settings


class ImageDetector:
    """Menjalankan deteksi kerusakan jalan pada satu citra."""

    def __init__(self, loader: ModelLoader) -> None:
        self.loader = loader

    @property
    def status(self) -> ModelStatus:
        return self.loader.status

    def detect(
        self,
        image_path: Union[str, Path],
        confidence: float = settings.DEFAULT_CONFIDENCE,
        iou: float = settings.DEFAULT_IOU,
        image_size: int = settings.DEFAULT_IMAGE_SIZE,
        device: Optional[str] = None,
        save_annotated: bool = False,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> DetectionResult:
        """
        Menjalankan deteksi pada satu citra.

        Parameter:
            image_path   : jalur ke citra input.
            confidence   : ambang keyakinan (0-1).
            iou          : ambang IoU untuk NMS.
            image_size   : ukuran input model (px).
            device       : 'cpu' / 'cuda:0' / None (default dari loader).
            save_annotated: simpan citra beranotasi jika True.
            output_dir   : folder tujuan anotasi (default outputs/detections).
        """
        image_path = Path(image_path)
        source_name = image_path.name

        # --- Validasi citra ---
        ok, msg = validate_image_file(image_path)
        if not ok:
            return DetectionResult(
                source=source_name, image_width=0, image_height=0, error=msg
            )

        # --- Validasi model ---
        if not self.loader.is_loaded():
            return DetectionResult(
                source=source_name, image_width=0, image_height=0,
                error="Model belum dimuat. Muat bobot di halaman Pengaturan Model.",
            )

        # --- Validasi parameter ---
        conf_ok, conf_msg, conf_val = validate_confidence(confidence)
        if not conf_ok:
            LOGGER.warning("Confidence: %s", conf_msg)
        size_ok, size_msg, size_val = validate_image_size(image_size)
        if not size_ok:
            LOGGER.warning("Image size: %s", size_msg)

        # --- Muat citra ---
        bgr = load_image(image_path)
        if bgr is None:
            return DetectionResult(
                source=source_name, image_width=0, image_height=0,
                error="Gagal memuat citra.",
            )
        h, w = bgr.shape[:2]

        # --- Inferensi ---
        try:
            model = self.loader.model
            infer_device = device or self.status.device
            t0 = time.perf_counter()
            results = model.predict(
                source=str(image_path),
                conf=conf_val,
                iou=iou,
                imgsz=size_val,
                device=infer_device,
                verbose=False,
            )
            inference_time = time.perf_counter() - t0
        except Exception as e:
            LOGGER.exception("Inferensi gagal untuk %s", image_path)
            return DetectionResult(
                source=source_name, image_width=w, image_height=h,
                error=f"Inferensi gagal: {e}",
            )

        # --- Ekstrak deteksi ---
        boxes = self._extract_detections(results, w, h)

        result = DetectionResult(
            source=source_name,
            image_width=w,
            image_height=h,
            detections=boxes,
            inference_time=inference_time,
        )

        # --- Simpan anotasi bila diminta ---
        if save_annotated and boxes is not None:
            annotated = annotate_image_bgr(bgr, boxes)
            if output_dir is None:
                output_dir = settings.OUTPUTS_DETECTION_DIR
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{safe_filename(image_path.stem)}_{timestamp_str()}.jpg"
            out_path = output_dir / out_name
            try:
                import cv2
                cv2.imwrite(str(out_path), annotated)
                result.annotated_path = str(out_path)
                LOGGER.info("Anotasi disimpan: %s", out_path)
            except Exception as e:
                LOGGER.warning("Gagal menyimpan anotasi: %s", e)

        return result

    def _extract_detections(
        self, results: list, img_w: int, img_h: int
    ) -> list:
        """Mengonversi output ultralytics menjadi list[DetectionBox]."""
        boxes: list = []
        class_names = self.status.class_names

        if not results:
            return boxes

        res = results[0]
        # --- Akses atribut secara defensif (kompatibilitas versi) ---
        try:
            box_obj = res.boxes
        except AttributeError:
            return boxes

        if box_obj is None:
            return boxes

        try:
            xyxy = box_obj.xyxy.cpu().numpy()  # (N, 4)
            confs = box_obj.conf.cpu().numpy()  # (N,)
            clss = box_obj.cls.cpu().numpy().astype(int)  # (N,)
        except AttributeError:
            # Fallback untuk struktur berbeda
            try:
                data = box_obj.data.cpu().numpy()  # (N, 6): x1,y1,x2,y2,conf,cls
                xyxy = data[:, :4]
                confs = data[:, 4]
                clss = data[:, 5].astype(int)
            except Exception as e:
                LOGGER.warning("Gagal mengekstrak box: %s", e)
                return boxes

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            cls_id = int(clss[i])
            cls_name = (
                class_names[cls_id]
                if 0 <= cls_id < len(class_names)
                else f"class_{cls_id}"
            )
            boxes.append(
                DetectionBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=float(confs[i]),
                    class_id=cls_id,
                    class_name=cls_name,
                )
            )
        return boxes


def detect_image_from_model(
    model_loader: ModelLoader,
    image_path: Union[str, Path],
    **kwargs,
) -> DetectionResult:
    """Fungsi pintas: jalankan deteksi citra dengan ModelLoader yang diberikan."""
    detector = ImageDetector(model_loader)
    return detector.detect(image_path, **kwargs)

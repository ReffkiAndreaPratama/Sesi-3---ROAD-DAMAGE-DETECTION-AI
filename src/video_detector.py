"""
Detektor video untuk RoadDamage AI — Bengkulu.

Memproses video frame-demi-frame dengan YOLO, menyimpan video hasil,
dan memberlakukan batas frame/durasi untuk mencegah penggunaan sumber
daya berlebihan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from src.model_loader import ModelLoader, ModelStatus
from src.utils import LOGGER, timestamp_str, safe_filename, validate_video_file
from src.visualization import annotate_image_bgr
from config import settings


@dataclass
class VideoDetectionSummary:
    """Ringkasan hasil deteksi video."""

    source: str
    total_frames: int = 0
    processed_frames: int = 0
    skipped_frames: int = 0
    total_detections: int = 0
    class_distribution: dict = field(default_factory=dict)
    duration_seconds: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    output_path: Optional[str] = None
    error: Optional[str] = None
    truncated: bool = False

    @property
    def is_error(self) -> bool:
        return self.error is not None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "total_detections": self.total_detections,
            "class_distribution": self.class_distribution,
            "duration_seconds": round(self.duration_seconds, 2),
            "fps": round(self.fps, 2),
            "width": self.width,
            "height": self.height,
            "output_path": self.output_path,
            "error": self.error,
            "truncated": self.truncated,
        }


class VideoDetector:
    """Menjalankan deteksi pada video."""

    def __init__(self, loader: ModelLoader) -> None:
        self.loader = loader

    @property
    def status(self) -> ModelStatus:
        return self.loader.status

    def detect(
        self,
        video_path: Union[str, Path],
        confidence: float = settings.DEFAULT_CONFIDENCE,
        iou: float = settings.DEFAULT_IOU,
        image_size: int = settings.DEFAULT_IMAGE_SIZE,
        max_frames: int = settings.MAX_VIDEO_FRAMES,
        max_duration: int = settings.MAX_VIDEO_DURATION_SECONDS,
        save_output: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        progress_callback=None,
    ) -> VideoDetectionSummary:
        """
        Memproses video dan menyimpan video beranotasi.

        progress_callback: callable(done, total) opsional.
        """
        video_path = Path(video_path)
        summary = VideoDetectionSummary(source=video_path.name)

        # --- Validasi video ---
        ok, msg = validate_video_file(video_path)
        if not ok:
            summary.error = msg
            return summary

        # --- Validasi model ---
        if not self.loader.is_loaded():
            summary.error = "Model belum dimuat. Muat bobot di halaman Pengaturan Model."
            return summary

        try:
            import cv2
        except ImportError:
            summary.error = "OpenCV (cv2) tidak tersedia."
            return summary

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            summary.error = (
                "Gagal membuka video. Codec atau format mungkin tidak didukung. "
                "Coba konversi ke MP4 (H.264) atau AVI."
            )
            return summary

        try:
            return self._process_video(
                cap, video_path, confidence, iou, image_size,
                max_frames, max_duration, save_output, output_dir,
                cv2, summary, progress_callback,
            )
        finally:
            cap.release()

    def _process_video(
        self, cap, video_path, confidence, iou, image_size,
        max_frames, max_duration, save_output, output_dir,
        cv2, summary, progress_callback,
    ) -> VideoDetectionSummary:
        model = self.loader.model
        class_names = self.status.class_names
        infer_device = self.status.device

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        summary.fps = float(fps)
        summary.total_frames = frame_count
        summary.width = width
        summary.height = height

        # Penulis output
        writer = None
        if save_output:
            if output_dir is None:
                output_dir = settings.OUTPUTS_VIDEO_DIR
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{safe_filename(video_path.stem)}_{timestamp_str()}.mp4"
            out_path = output_dir / out_name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            if width and height:
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
                summary.output_path = str(out_path)
            else:
                LOGGER.warning("Dimensi video tidak diketahui; output tidak disimpan.")

        t_start = time.perf_counter()
        frame_idx = 0
        max_duration_frames = int(max_duration * fps) if fps > 0 else max_duration
        effective_max = min(max_frames, max_duration_frames) if max_duration_frames else max_frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx > effective_max:
                summary.truncated = True
                LOGGER.info(
                    "Video dipotong pada frame %d (batas frame/durasi).", effective_max
                )
                break

            try:
                results = model.predict(
                    source=frame,
                    conf=confidence,
                    iou=iou,
                    imgsz=image_size,
                    device=infer_device,
                    verbose=False,
                )
            except Exception as e:
                LOGGER.warning("Inferensi gagal pada frame %d: %s", frame_idx, e)
                summary.skipped_frames += 1
                continue

            boxes = self._extract_boxes(results, class_names)
            summary.total_detections += len(boxes)
            for b in boxes:
                summary.class_distribution[b.class_name] = (
                    summary.class_distribution.get(b.class_name, 0) + 1
                )

            if writer is not None:
                annotated = annotate_image_bgr(frame, boxes)
                writer.write(annotated)

            summary.processed_frames += 1

            if progress_callback is not None and frame_idx % 5 == 0:
                try:
                    progress_callback(frame_idx, effective_max)
                except Exception:
                    pass

        if writer is not None:
            writer.release()

        summary.duration_seconds = time.perf_counter() - t_start
        LOGGER.info(
            "Video selesai: %d frame diproses, %d deteksi, %.2fs",
            summary.processed_frames, summary.total_detections, summary.duration_seconds,
        )
        return summary

    @staticmethod
    def _extract_boxes(results: list, class_names: List[str]):
        from src.utils import DetectionBox

        boxes = []
        if not results:
            return boxes
        res = results[0]
        try:
            box_obj = res.boxes
        except AttributeError:
            return boxes
        if box_obj is None:
            return boxes
        try:
            xyxy = box_obj.xyxy.cpu().numpy()
            confs = box_obj.conf.cpu().numpy()
            clss = box_obj.cls.cpu().numpy().astype(int)
        except AttributeError:
            try:
                data = box_obj.data.cpu().numpy()
                xyxy = data[:, :4]
                confs = data[:, 4]
                clss = data[:, 5].astype(int)
            except Exception:
                return boxes

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            cls_id = int(clss[i])
            name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
            boxes.append(DetectionBox(
                x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                confidence=float(confs[i]), class_id=cls_id, class_name=name,
            ))
        return boxes

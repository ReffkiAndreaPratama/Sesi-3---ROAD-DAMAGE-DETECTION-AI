"""
Pengujian untuk src/image_detector.py dan src/utils.py (validasi + ekspor).

Mencakup:
- Deteksi pada citra valid menggunakan mock loader
- Penanganan citra rusak / tidak ada
- Model belum dimuat
- Validasi confidence threshold
- Format dan isi DetectionResult
- Ekspor CSV dan JSON
- Pengujian batch (BatchProcessor)
- Penanganan batch kosong
- Distribusi kelas dari hasil deteksi
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.image_detector import ImageDetector, detect_image_from_model
from src.batch_processor import BatchProcessor, validate_batch_files
from src.utils import (
    DetectionBox,
    DetectionResult,
    export_results_csv,
    export_results_json,
    validate_confidence,
    validate_image_file,
    validate_image_size,
    load_image,
    load_image_rgb,
    safe_filename,
    timestamp_str,
    format_size,
    is_cuda_available,
)


# ============================================================
# Tes validasi citra
# ============================================================
class TestValidateImageFile:
    def test_valid_png(self, valid_image_path: Path):
        ok, msg = validate_image_file(valid_image_path)
        assert ok, f"Harus valid: {msg}"

    def test_valid_jpg(self, valid_jpg_path: Path):
        ok, msg = validate_image_file(valid_jpg_path)
        assert ok, f"Harus valid: {msg}"

    def test_corrupted_image_rejected(self, corrupted_image_path: Path):
        ok, msg = validate_image_file(corrupted_image_path)
        assert not ok
        assert msg  # pesan error tidak boleh kosong

    def test_empty_file_rejected(self, empty_image_path: Path):
        ok, msg = validate_image_file(empty_image_path)
        assert not ok

    def test_file_not_found(self, tmp_path: Path):
        ok, msg = validate_image_file(tmp_path / "ghost.png")
        assert not ok
        assert "tidak ditemukan" in msg.lower() or "not found" in msg.lower()

    def test_unsupported_extension(self, tmp_path: Path):
        p = tmp_path / "document.bmp"
        p.write_bytes(b"data")
        ok, msg = validate_image_file(p)
        assert not ok
        assert "tidak didukung" in msg.lower() or "not supported" in msg.lower()


# ============================================================
# Tes muat citra
# ============================================================
class TestLoadImage:
    def test_load_valid_image_returns_array(self, valid_image_path: Path):
        arr = load_image(valid_image_path)
        assert arr is not None
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 3

    def test_load_rgb_returns_array(self, valid_image_path: Path):
        arr = load_image_rgb(valid_image_path)
        assert arr is not None
        assert arr.shape[2] == 3

    def test_load_corrupted_returns_none(self, corrupted_image_path: Path):
        arr = load_image(corrupted_image_path)
        assert arr is None


# ============================================================
# Tes ImageDetector
# ============================================================
class TestImageDetector:
    def test_detect_returns_result(self, valid_image_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(valid_image_path, confidence=0.5)
        assert isinstance(result, DetectionResult)
        assert not result.is_error

    def test_detect_has_correct_num_detections(self, valid_image_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(valid_image_path)
        # mock mengembalikan 1 box
        assert result.num_detections == 1

    def test_detection_box_class_name(self, valid_image_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(valid_image_path)
        assert result.detections[0].class_name == "D00"

    def test_detection_box_confidence_range(self, valid_image_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(valid_image_path)
        for d in result.detections:
            assert 0.0 <= d.confidence <= 1.0

    def test_detect_corrupted_image(self, corrupted_image_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(corrupted_image_path)
        assert result.is_error

    def test_detect_model_not_loaded(self, valid_image_path: Path, mock_loader_not_loaded):
        detector = ImageDetector(mock_loader_not_loaded)
        result = detector.detect(valid_image_path)
        assert result.is_error
        assert result.error  # pesan error tidak boleh kosong

    def test_detect_file_not_found(self, tmp_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(tmp_path / "nonexistent.jpg")
        assert result.is_error

    def test_detect_records_inference_time(self, valid_image_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(valid_image_path)
        assert result.inference_time >= 0.0

    def test_detect_result_to_dict(self, valid_image_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(valid_image_path)
        d = result.to_dict()
        for key in ("source", "image_width", "image_height", "num_detections",
                    "inference_time", "detections", "class_distribution"):
            assert key in d

    def test_class_distribution(self, valid_image_path: Path, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        result = detector.detect(valid_image_path)
        dist = result.class_distribution()
        assert isinstance(dist, dict)
        assert "D00" in dist

    def test_shortcut_function(self, valid_image_path: Path, mock_loader_loaded):
        result = detect_image_from_model(mock_loader_loaded, valid_image_path)
        assert isinstance(result, DetectionResult)


# ============================================================
# Tes validasi parameter
# ============================================================
class TestValidateConfidence:
    @pytest.mark.parametrize("conf,expected_valid", [
        (0.25, True),
        (0.0,  True),
        (1.0,  True),
        (0.5,  True),
        (-0.1, False),
        (1.5,  False),
    ])
    def test_confidence_range(self, conf, expected_valid):
        valid, msg, _ = validate_confidence(conf)
        assert valid == expected_valid

    def test_none_confidence_uses_default(self):
        from config import settings
        valid, msg, val = validate_confidence(None)
        assert valid
        assert val == settings.DEFAULT_CONFIDENCE

    def test_string_confidence_fails(self):
        valid, msg, _ = validate_confidence("bukan angka")
        assert not valid


class TestValidateImageSize:
    @pytest.mark.parametrize("size,expected_valid", [
        (640, True),
        (320, True),
        (1024, True),
        (0, False),
        (-1, False),
    ])
    def test_size_range(self, size, expected_valid):
        valid, msg, _ = validate_image_size(size)
        assert valid == expected_valid


# ============================================================
# Tes DetectionBox
# ============================================================
class TestDetectionBox:
    def test_properties(self):
        box = DetectionBox(x1=10, y1=20, x2=110, y2=120,
                           confidence=0.9, class_id=0, class_name="D00")
        assert box.width == 100
        assert box.height == 100
        assert box.area == 10000

    def test_to_dict(self):
        box = DetectionBox(x1=10, y1=20, x2=110, y2=120,
                           confidence=0.9, class_id=0, class_name="D00")
        d = box.to_dict()
        assert d["class_name"] == "D00"
        assert d["confidence"] == 0.9

    def test_to_label_dict_normalizes(self):
        box = DetectionBox(x1=0, y1=0, x2=64, y2=64,
                           confidence=0.8, class_id=0, class_name="D00")
        ld = box.to_label_dict(img_w=640, img_h=640)
        assert 0.0 <= ld["cx"] <= 1.0
        assert 0.0 <= ld["cy"] <= 1.0
        assert 0.0 <= ld["w"] <= 1.0


# ============================================================
# Tes ekspor CSV dan JSON
# ============================================================
class TestExportResults:
    def _make_result(self, source="img.jpg", n_dets=2):
        boxes = [
            DetectionBox(10, 20, 100, 80, 0.85, 0, "D00"),
            DetectionBox(50, 60, 200, 180, 0.72, 2, "D10"),
        ][:n_dets]
        return DetectionResult(
            source=source, image_width=640, image_height=480,
            detections=boxes, inference_time=0.05,
        )

    def test_export_csv_creates_file(self, tmp_path: Path):
        result = self._make_result()
        out = tmp_path / "test_results.csv"
        path = export_results_csv([result], out)
        assert path.exists()

    def test_export_csv_has_header(self, tmp_path: Path):
        result = self._make_result()
        out = tmp_path / "test.csv"
        export_results_csv([result], out)
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert "source" in headers
        assert "class_name" in headers
        assert "confidence" in headers

    def test_export_csv_row_count(self, tmp_path: Path):
        results = [self._make_result(f"img_{i}.jpg", n_dets=2) for i in range(3)]
        out = tmp_path / "batch.csv"
        export_results_csv(results, out)
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        # 3 gambar × 2 deteksi = 6 baris
        assert len(rows) == 6

    def test_export_csv_no_detections(self, tmp_path: Path):
        result = DetectionResult(
            source="empty.jpg", image_width=640, image_height=480,
            detections=[], inference_time=0.01,
        )
        out = tmp_path / "empty.csv"
        export_results_csv([result], out)
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["class_name"] == "(tidak ada deteksi)"

    def test_export_json_creates_file(self, tmp_path: Path):
        result = self._make_result()
        out = tmp_path / "test.json"
        export_results_json([result], out)
        assert out.exists()

    def test_export_json_structure(self, tmp_path: Path):
        result = self._make_result()
        out = tmp_path / "test.json"
        export_results_json([result], out)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert "results" in data
        assert "total_images" in data
        assert data["total_images"] == 1
        assert data["total_detections"] == 2

    def test_export_json_error_result(self, tmp_path: Path):
        result = DetectionResult(
            source="bad.jpg", image_width=0, image_height=0,
            error="Citra rusak",
        )
        out = tmp_path / "error.json"
        export_results_json([result], out)
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert data["results"][0]["error"] == "Citra rusak"


# ============================================================
# Tes BatchProcessor
# ============================================================
class TestBatchProcessor:
    def test_empty_batch(self, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        batch = BatchProcessor(detector)
        results, progress = batch.process([])
        assert results == []
        assert progress.done == 0
        assert progress.total == 0

    def test_batch_processes_all_valid_images(self, batch_image_paths, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        batch = BatchProcessor(detector)
        results, progress = batch.process(batch_image_paths)
        assert len(results) == len(batch_image_paths)
        assert progress.done == len(batch_image_paths)

    def test_batch_summary_table_has_rows(self, batch_image_paths, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        batch = BatchProcessor(detector)
        batch.process(batch_image_paths)
        table = batch.summary_table()
        assert len(table) == len(batch_image_paths)

    def test_batch_handles_mixed_valid_invalid(self, tmp_path, mock_loader_loaded):
        # 2 citra valid + 1 file tidak ada
        from PIL import Image
        valid = []
        for i in range(2):
            p = tmp_path / f"valid_{i}.png"
            Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(p)
            valid.append(p)
        ghost = tmp_path / "ghost.png"  # tidak ada
        all_paths = valid + [ghost]

        detector = ImageDetector(mock_loader_loaded)
        batch = BatchProcessor(detector)
        results, progress = batch.process(all_paths)
        # ghost akan di-detect dengan error
        errors = [r for r in results if r.is_error]
        assert len(errors) >= 1

    def test_batch_aggregate_distribution(self, batch_image_paths, mock_loader_loaded):
        detector = ImageDetector(mock_loader_loaded)
        batch = BatchProcessor(detector)
        batch.process(batch_image_paths)
        dist = batch.aggregate_distribution()
        assert isinstance(dist, dict)
        # Mock mengembalikan 1 deteksi "D00" per citra
        assert dist.get("D00", 0) == len(batch_image_paths)

    def test_batch_export_csv(self, batch_image_paths, mock_loader_loaded, tmp_path):
        detector = ImageDetector(mock_loader_loaded)
        batch = BatchProcessor(detector)
        batch.process(batch_image_paths)
        out = tmp_path / "batch.csv"
        path = batch.export_csv(out)
        assert path.exists()

    def test_batch_export_json(self, batch_image_paths, mock_loader_loaded, tmp_path):
        detector = ImageDetector(mock_loader_loaded)
        batch = BatchProcessor(detector)
        batch.process(batch_image_paths)
        out = tmp_path / "batch.json"
        path = batch.export_json(out)
        assert path.exists()

    def test_batch_progress_callback_called(self, batch_image_paths, mock_loader_loaded):
        calls = []
        def cb(p):
            calls.append(p.done)

        detector = ImageDetector(mock_loader_loaded)
        batch = BatchProcessor(detector)
        batch.process(batch_image_paths, callback=cb)
        assert len(calls) == len(batch_image_paths)


# ============================================================
# Tes validate_batch_files
# ============================================================
class TestValidateBatchFiles:
    def test_all_valid(self, batch_image_paths):
        valid, errs = validate_batch_files(batch_image_paths)
        assert len(valid) == len(batch_image_paths)
        assert len(errs) == 0

    def test_mix_valid_invalid(self, tmp_path, batch_image_paths):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"bukan citra")
        all_paths = batch_image_paths + [bad]
        valid, errs = validate_batch_files(all_paths)
        assert len(valid) == len(batch_image_paths)
        assert len(errs) == 1


# ============================================================
# Tes utilitas umum
# ============================================================
class TestUtilities:
    def test_safe_filename_removes_special_chars(self):
        name = safe_filename("jalan/rusak:berat?", ".jpg")
        assert "/" not in name
        assert ":" not in name
        assert name.endswith(".jpg")

    def test_timestamp_str_nonempty(self):
        ts = timestamp_str()
        assert len(ts) > 0
        assert "_" in ts

    def test_format_size_kb(self):
        result = format_size(2048)
        assert "KB" in result or "B" in result

    def test_format_size_mb(self):
        result = format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_is_cuda_available_returns_bool(self):
        result = is_cuda_available()
        assert isinstance(result, bool)

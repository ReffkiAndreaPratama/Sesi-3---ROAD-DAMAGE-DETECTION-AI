"""
Pengujian untuk src/dataset_utils.py

Mencakup:
- Konversi XML RDD ke format YOLO
- Penanganan kelas tidak dikenal dalam XML
- Penanganan XML rusak
- Normalisasi koordinat bounding box
- Validasi struktur dataset YOLO
- Pembuatan dan pembacaan data.yaml
- Laporan dataset (DatasetReport)
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image
import numpy as np

from src.dataset_utils import (
    DatasetReport,
    _scan_label_file,
    convert_rdd_dataset,
    get_default_class_list,
    get_rdd_class_names,
    rdd_xml_to_yolo,
    read_data_yaml,
    validate_yolo_dataset,
    write_data_yaml,
    get_dataset_instructions,
)


# ============================================================
# Tes pemetaan kelas
# ============================================================
class TestClassMapping:
    def test_get_rdd_class_names_returns_dict(self):
        names = get_rdd_class_names()
        assert isinstance(names, dict)
        assert "D00" in names
        assert "D40" in names

    def test_get_default_class_list_returns_list(self):
        classes = get_default_class_list()
        assert isinstance(classes, list)
        assert len(classes) > 0
        assert "D00" in classes


# ============================================================
# Tes rdd_xml_to_yolo
# ============================================================
class TestRddXmlToYolo:
    def test_valid_xml_produces_correct_lines(self, valid_rdd_xml: Path):
        lines = rdd_xml_to_yolo(valid_rdd_xml, image_width=640, image_height=480)
        assert len(lines) == 2  # 2 objek di fixture

    def test_yolo_line_format(self, valid_rdd_xml: Path):
        lines = rdd_xml_to_yolo(valid_rdd_xml, image_width=640, image_height=480)
        for line in lines:
            parts = line.split()
            assert len(parts) == 5, f"Format salah: {line}"
            # Koordinat normalisasi 0-1
            vals = [float(p) for p in parts[1:]]
            for v in vals:
                assert 0.0 <= v <= 1.0, f"Koordinat di luar [0,1]: {v}"

    def test_class_id_correct(self, valid_rdd_xml: Path):
        class_id_map = {c: i for i, c in enumerate(get_default_class_list())}
        lines = rdd_xml_to_yolo(valid_rdd_xml, 640, 480, class_id_map)
        cls_ids = [int(l.split()[0]) for l in lines]
        assert 0 in cls_ids  # D00 → 0

    def test_unknown_class_skipped(self, unknown_class_rdd_xml: Path):
        lines = rdd_xml_to_yolo(unknown_class_rdd_xml, 640, 480)
        # Kelas tidak dikenal harus dilewati
        assert len(lines) == 0

    def test_zero_image_size_raises(self, valid_rdd_xml: Path):
        with pytest.raises(ValueError):
            rdd_xml_to_yolo(valid_rdd_xml, image_width=0, image_height=480)

    def test_invalid_xml_raises(self, tmp_path: Path):
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("ini bukan xml yang valid <<>>!!", encoding="utf-8")
        with pytest.raises(ValueError):
            rdd_xml_to_yolo(bad_xml, 640, 480)

    def test_normalized_cx_cy_within_bounds(self, valid_rdd_xml: Path):
        lines = rdd_xml_to_yolo(valid_rdd_xml, 640, 480)
        for line in lines:
            parts = line.split()
            cx, cy = float(parts[1]), float(parts[2])
            assert 0.0 < cx < 1.0, f"cx di luar (0,1): {cx}"
            assert 0.0 < cy < 1.0, f"cy di luar (0,1): {cy}"


# ============================================================
# Tes convert_rdd_dataset
# ============================================================
class TestConvertRddDataset:
    def test_converts_valid_xml_files(self, tmp_path: Path, valid_rdd_xml: Path):
        """Konversi XML yang valid di folder harus menghasilkan file .txt."""
        annotation_dir = tmp_path / "annotations"
        annotation_dir.mkdir()
        # Salin fixture ke annotation_dir
        import shutil
        shutil.copy(valid_rdd_xml, annotation_dir / valid_rdd_xml.name)

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # Buat citra pendamping agar ukuran dibaca dari XML
        Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(
            img_dir / "road_001.jpg"
        )

        out_dir = tmp_path / "labels"
        success, fail, errors = convert_rdd_dataset(annotation_dir, img_dir, out_dir)
        assert success == 1
        assert fail == 0
        assert (out_dir / "road_001.txt").exists()

    def test_missing_annotation_dir_returns_error(self, tmp_path: Path):
        success, fail, errors = convert_rdd_dataset(
            tmp_path / "nonexistent", tmp_path, tmp_path / "out"
        )
        assert success == 0
        assert errors

    def test_empty_annotation_dir(self, tmp_path: Path):
        (tmp_path / "anns").mkdir()
        (tmp_path / "imgs").mkdir()
        success, fail, errors = convert_rdd_dataset(
            tmp_path / "anns", tmp_path / "imgs", tmp_path / "out"
        )
        assert success == 0
        assert fail == 0


# ============================================================
# Tes validasi label YOLO (_scan_label_file)
# ============================================================
class TestScanLabelFile:
    def test_valid_labels_counted_in_distribution(self, valid_label_file: Path):
        report = DatasetReport()
        _scan_label_file(valid_label_file, report)
        assert report.invalid_label_lines == 0
        assert len(report.class_distribution) > 0

    def test_invalid_lines_counted(self, invalid_label_file: Path):
        report = DatasetReport()
        _scan_label_file(invalid_label_file, report)
        assert report.invalid_label_lines > 0

    def test_empty_label_counted(self, empty_label_file: Path):
        report = DatasetReport()
        _scan_label_file(empty_label_file, report)
        assert report.empty_label_files == 1
        assert report.invalid_label_lines == 0


# ============================================================
# Tes validate_yolo_dataset
# ============================================================
class TestValidateYoloDataset:
    def _create_mini_dataset(self, root: Path, n_train=3, n_val=2, n_test=1):
        """Membuat dataset YOLO mini di tmp."""
        for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
            img_dir = root / "images" / split
            lbl_dir = root / "labels" / split
            img_dir.mkdir(parents=True)
            lbl_dir.mkdir(parents=True)
            for i in range(n):
                stem = f"{split}_{i}"
                Image.fromarray(np.zeros((80, 80, 3), dtype=np.uint8)).save(
                    img_dir / f"{stem}.jpg"
                )
                (lbl_dir / f"{stem}.txt").write_text(
                    "0 0.5 0.5 0.3 0.4\n", encoding="utf-8"
                )

    def test_empty_dir_not_found(self, tmp_path: Path):
        report = validate_yolo_dataset(tmp_path / "empty")
        assert not report.found

    def test_valid_dataset_found(self, tmp_path: Path):
        self._create_mini_dataset(tmp_path)
        report = validate_yolo_dataset(tmp_path)
        assert report.found

    def test_correct_image_counts(self, tmp_path: Path):
        self._create_mini_dataset(tmp_path, n_train=3, n_val=2, n_test=1)
        report = validate_yolo_dataset(tmp_path)
        assert report.images_per_split.get("train", 0) == 3
        assert report.images_per_split.get("val", 0) == 2
        assert report.images_per_split.get("test", 0) == 1
        assert report.total_images == 6

    def test_missing_label_detected(self, tmp_path: Path):
        self._create_mini_dataset(tmp_path, n_train=2, n_val=0, n_test=0)
        # Hapus satu file label
        lbl = list((tmp_path / "labels" / "train").glob("*.txt"))[0]
        lbl.unlink()
        report = validate_yolo_dataset(tmp_path)
        assert len(report.missing_labels) >= 1

    def test_to_dict_has_required_keys(self, tmp_path: Path):
        self._create_mini_dataset(tmp_path)
        report = validate_yolo_dataset(tmp_path)
        d = report.to_dict()
        for key in ("found", "images_per_split", "total_images", "errors"):
            assert key in d


# ============================================================
# Tes write_data_yaml dan read_data_yaml
# ============================================================
class TestDataYaml:
    def test_write_creates_file(self, tmp_path: Path):
        p = write_data_yaml(tmp_path, class_names=["D00", "D10"])
        assert p.exists()

    def test_written_yaml_contains_path(self, tmp_path: Path):
        p = write_data_yaml(tmp_path, class_names=["D00"])
        content = p.read_text(encoding="utf-8")
        assert "images/train" in content
        assert "D00" in content

    def test_read_nonexistent_returns_none(self, tmp_path: Path):
        result = read_data_yaml(tmp_path / "ghost.yaml")
        assert result is None

    def test_read_written_yaml(self, tmp_path: Path):
        write_data_yaml(tmp_path, class_names=["D00", "D10", "D40"])
        data = read_data_yaml(tmp_path / "data.yaml")
        assert data is not None
        # names bisa berupa dict {0: 'D00', ...}
        names = data.get("names", {})
        if isinstance(names, dict):
            values = list(names.values())
        else:
            values = list(names)
        assert "D00" in values
        assert "D40" in values

    def test_write_custom_output_path(self, tmp_path: Path):
        out = tmp_path / "custom_dir" / "my_data.yaml"
        p = write_data_yaml(tmp_path, class_names=["D00"], output_path=out)
        assert p.exists()

    def test_write_default_classes(self, tmp_path: Path):
        p = write_data_yaml(tmp_path)
        content = p.read_text(encoding="utf-8")
        assert "D00" in content or "D10" in content


# ============================================================
# Tes panduan dataset
# ============================================================
class TestDatasetInstructions:
    def test_instructions_nonempty(self):
        text = get_dataset_instructions()
        assert len(text) > 100

    def test_instructions_mentions_rdd(self):
        text = get_dataset_instructions()
        assert "RDD" in text or "sekilab" in text or "Road Damage" in text

    def test_instructions_mentions_yolo_format(self):
        text = get_dataset_instructions()
        assert "YOLO" in text or "images" in text

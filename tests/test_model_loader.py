"""
Pengujian untuk src/model_loader.py

Mencakup:
- Pemuatan bobot yang tidak ada
- Deteksi perangkat (CPU/GPU mock)
- Klasifikasi jenis model (kerusakan jalan vs. umum)
- Ekstraksi nama kelas dari dict dan list
- Peringatan untuk model umum
- Helper list_available_weights dan check_weights_exists
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.model_loader import (
    ModelLoader,
    ModelStatus,
    check_weights_exists,
    get_pretrained_options,
    list_available_weights,
)
from config import settings


# ============================================================
# Helper
# ============================================================
def _make_mock_yolo(class_names: dict | list):
    """Membuat mock objek YOLO dengan names tertentu."""
    m = MagicMock()
    m.names = class_names
    m.to.return_value = m
    return m


# ============================================================
# Tes pemuatan bobot tidak ada
# ============================================================
class TestWeightsNotFound:
    def test_missing_file_returns_error_status(self, tmp_path: Path):
        """Bobot tidak ada di disk harus menghasilkan error, bukan exception."""
        loader = ModelLoader()
        fake_path = tmp_path / "not_existing.pt"
        status = loader.load(fake_path)
        assert not status.loaded
        assert status.errors
        # Pesan error bisa berisi "tidak ditemukan" (file) ATAU
        # "ultralytics" (jika belum terpasang) — keduanya valid.
        combined = " ".join(e.lower() for e in status.errors)
        assert (
            "tidak ditemukan" in combined
            or "tidak ada" in combined
            or "ultralytics" in combined
            or "terpasang" in combined
        ), f"Pesan error tidak dikenali: {status.errors}"

    def test_missing_file_model_not_loaded(self, tmp_path: Path):
        loader = ModelLoader()
        status = loader.load(tmp_path / "ghost.pt")
        assert not loader.is_loaded()

    def test_access_model_before_load_raises(self, tmp_path: Path):
        from src.model_loader import ModelLoadError
        loader = ModelLoader()
        with pytest.raises(ModelLoadError):
            _ = loader.model


# ============================================================
# Tes saat ultralytics tidak tersedia
# ============================================================
class TestUltralyticsMissing:
    def test_no_ultralytics_returns_error(self, tmp_path: Path):
        """Jika ultralytics tidak terpasang, status harus error."""
        import src.model_loader as ml_mod
        original = ml_mod.YOLO
        try:
            ml_mod.YOLO = None  # Simulasikan ultralytics tidak tersedia
            loader = ModelLoader()
            p = tmp_path / "dummy.pt"
            p.write_bytes(b"dummy")
            status = loader.load(p)
            assert not status.loaded
            assert any("ultralytics" in e.lower() for e in status.errors)
        finally:
            ml_mod.YOLO = original


# ============================================================
# Tes pemuatan berhasil (mock YOLO)
# ============================================================
class TestSuccessfulLoad:
    def test_load_road_damage_weights(self, tmp_path: Path):
        """Bobot dengan nama kelas kerusakan jalan harus menghasilkan model_type='road_damage'."""
        fake_pt = tmp_path / "road_damage_best.pt"
        fake_pt.write_bytes(b"fake")

        mock_model = _make_mock_yolo({0: "D00", 1: "D10", 2: "D40"})

        with patch("src.model_loader.YOLO", return_value=mock_model):
            loader = ModelLoader()
            loader._ultralytics_ok = True
            status = loader.load(fake_pt)

        assert status.loaded
        assert status.model_type == "road_damage"
        assert status.num_classes == 3
        assert "D00" in status.class_names

    def test_load_general_weights_shows_warning(self, tmp_path: Path):
        """Bobot dengan nama pralatih umum harus model_type='general' + warning."""
        mock_model = _make_mock_yolo(list(range(80)))

        with patch("src.model_loader.YOLO", return_value=mock_model):
            loader = ModelLoader()
            loader._ultralytics_ok = True
            status = loader.load("yolov8n.pt")

        assert status.model_type == "general"
        assert status.warnings
        assert any("bukan" in w.lower() or "coco" in w.lower() or "umum" in w.lower()
                   for w in status.warnings)

    def test_class_names_from_dict(self, tmp_path: Path):
        """Nama kelas dari dict harus diurutkan berdasar kunci."""
        fake_pt = tmp_path / "rdd_model.pt"
        fake_pt.write_bytes(b"x")
        names_dict = {0: "D00", 2: "D20", 1: "D10"}
        mock_model = _make_mock_yolo(names_dict)

        with patch("src.model_loader.YOLO", return_value=mock_model):
            loader = ModelLoader()
            loader._ultralytics_ok = True
            status = loader.load(fake_pt)

        assert status.class_names == ["D00", "D10", "D20"]

    def test_class_names_from_list(self, tmp_path: Path):
        fake_pt = tmp_path / "model_list.pt"
        fake_pt.write_bytes(b"x")
        mock_model = _make_mock_yolo(["D00", "D10", "D40"])

        with patch("src.model_loader.YOLO", return_value=mock_model):
            loader = ModelLoader()
            loader._ultralytics_ok = True
            status = loader.load(fake_pt)

        assert status.class_names == ["D00", "D10", "D40"]

    def test_device_defaults_to_cpu(self, tmp_path: Path):
        fake_pt = tmp_path / "model.pt"
        fake_pt.write_bytes(b"x")
        mock_model = _make_mock_yolo(["D00"])

        with patch("src.model_loader.YOLO", return_value=mock_model):
            loader = ModelLoader(prefer_gpu=False)
            loader._ultralytics_ok = True
            status = loader.load(fake_pt)

        assert status.device == "cpu"

    def test_reload_resets_state(self, tmp_path: Path):
        """Memuat ulang model harus mengatur ulang status sebelumnya."""
        fake_pt = tmp_path / "model.pt"
        fake_pt.write_bytes(b"x")
        mock_model = _make_mock_yolo(["D00"])

        with patch("src.model_loader.YOLO", return_value=mock_model):
            loader = ModelLoader()
            loader._ultralytics_ok = True
            status1 = loader.load(fake_pt)
        assert status1.loaded

        # Muat ulang dengan file tidak ada -> harus error
        status2 = loader.load(tmp_path / "nonexistent.pt")
        assert not status2.loaded


# ============================================================
# Tes klasifikasi model
# ============================================================
class TestModelClassification:
    @pytest.mark.parametrize("class_names,expected", [
        (["D00", "D10", "D40"], "road_damage"),
        (["crack", "pothole", "rut"], "road_damage"),
        (["person", "car", "truck"], "general"),
        ([], "unknown"),
    ])
    def test_classify_model_type(self, tmp_path: Path, class_names, expected):
        fake_pt = tmp_path / f"model_{expected}.pt"
        fake_pt.write_bytes(b"x")
        mock_model = _make_mock_yolo(class_names)

        with patch("src.model_loader.YOLO", return_value=mock_model):
            loader = ModelLoader()
            loader._ultralytics_ok = True
            status = loader.load(fake_pt)

        assert status.model_type == expected, (
            f"Kelas {class_names} diharapkan '{expected}', dapat '{status.model_type}'"
        )


# ============================================================
# Tes helper publik
# ============================================================
class TestHelpers:
    def test_list_weights_empty_dir(self, tmp_path: Path):
        weights = list_available_weights(tmp_path)
        assert weights == []

    def test_list_weights_finds_pt_files(self, tmp_path: Path):
        (tmp_path / "best.pt").write_bytes(b"x")
        (tmp_path / "last.pt").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("bukan bobot")
        weights = list_available_weights(tmp_path)
        assert sorted(weights) == ["best.pt", "last.pt"]

    def test_check_weights_exists_found(self, tmp_path: Path):
        p = tmp_path / "model.pt"
        p.write_bytes(b"x")
        ok, msg = check_weights_exists(p)
        assert ok

    def test_check_weights_not_found(self, tmp_path: Path):
        ok, msg = check_weights_exists(tmp_path / "ghost.pt")
        assert not ok
        assert "tidak ditemukan" in msg.lower() or "tidak ada" in msg.lower()

    def test_check_pretrained_name_is_valid(self):
        """Nama pralatih seperti 'yolov8n.pt' harus dianggap valid."""
        ok, msg = check_weights_exists("yolov8n.pt")
        assert ok

    def test_get_pretrained_options_returns_dict(self):
        opts = get_pretrained_options()
        assert isinstance(opts, dict)
        assert "yolov8n.pt" in opts or len(opts) > 0

    def test_model_status_to_dict_keys(self):
        status = ModelStatus(loaded=True, device="cpu", num_classes=3)
        d = status.to_dict()
        for key in ("loaded", "device", "num_classes", "class_names", "warnings", "errors"):
            assert key in d

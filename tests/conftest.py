"""
Fixtures bersama untuk seluruh pengujian RoadDamage AI — Bengkulu.

Semua fixture yang melibatkan citra, file label, dan model menggunakan
data kecil yang dibuat secara programatik agar pengujian tidak memerlukan
bobot YOLO atau dataset nyata.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest
from PIL import Image

# Pastikan root proyek ada di sys.path saat pytest dijalankan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Fixture: file citra
# ============================================================
@pytest.fixture
def valid_image_path(tmp_path: Path) -> Path:
    """Membuat file PNG valid 100×100 pixel untuk pengujian."""
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    p = tmp_path / "test_road.png"
    img.save(p)
    return p


@pytest.fixture
def valid_jpg_path(tmp_path: Path) -> Path:
    """Membuat file JPEG valid 200×150 pixel."""
    arr = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    p = tmp_path / "test_road.jpg"
    img.save(p)
    return p


@pytest.fixture
def corrupted_image_path(tmp_path: Path) -> Path:
    """Membuat file PNG rusak (isi bukan data citra)."""
    p = tmp_path / "bad_image.png"
    p.write_bytes(b"bukan data citra sama sekali!!")
    return p


@pytest.fixture
def empty_image_path(tmp_path: Path) -> Path:
    """Membuat file PNG kosong (0 byte)."""
    p = tmp_path / "empty.png"
    p.write_bytes(b"")
    return p


@pytest.fixture
def batch_image_paths(tmp_path: Path) -> list:
    """Membuat 5 citra PNG valid untuk pengujian batch."""
    paths = []
    for i in range(5):
        arr = np.zeros((80, 80, 3), dtype=np.uint8)
        arr[:, :, i % 3] = 128  # variasikan warna
        img = Image.fromarray(arr)
        p = tmp_path / f"batch_{i}.png"
        img.save(p)
        paths.append(p)
    return paths


# ============================================================
# Fixture: file label YOLO
# ============================================================
@pytest.fixture
def valid_label_file(tmp_path: Path) -> Path:
    """Membuat file label YOLO valid (kelas 0, koordinat normalisasi)."""
    p = tmp_path / "sample.txt"
    p.write_text("0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.2\n", encoding="utf-8")
    return p


@pytest.fixture
def invalid_label_file(tmp_path: Path) -> Path:
    """Membuat file label YOLO berisi baris tidak valid."""
    p = tmp_path / "bad_labels.txt"
    p.write_text(
        "bukan angka di sini\n"
        "0 0.5 0.5 0.3\n"       # kurang satu kolom
        "0 1.5 0.5 0.3 0.4\n",  # koordinat > 1
        encoding="utf-8",
    )
    return p


@pytest.fixture
def empty_label_file(tmp_path: Path) -> Path:
    """Membuat file label YOLO kosong."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


# ============================================================
# Fixture: data XML RDD
# ============================================================
@pytest.fixture
def valid_rdd_xml(tmp_path: Path) -> Path:
    """Membuat file XML RDD minimal yang valid."""
    xml_content = """\
<?xml version="1.0" encoding="UTF-8"?>
<annotation>
  <filename>road_001.jpg</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>D00</name>
    <bndbox>
      <xmin>50</xmin><ymin>60</ymin><xmax>200</xmax><ymax>150</ymax>
    </bndbox>
  </object>
  <object>
    <name>D10</name>
    <bndbox>
      <xmin>300</xmin><ymin>100</ymin><xmax>450</xmax><ymax>200</ymax>
    </bndbox>
  </object>
</annotation>
"""
    p = tmp_path / "road_001.xml"
    p.write_text(xml_content, encoding="utf-8")
    return p


@pytest.fixture
def unknown_class_rdd_xml(tmp_path: Path) -> Path:
    """Membuat file XML RDD dengan kelas yang tidak ada di pemetaan."""
    xml_content = """\
<?xml version="1.0" encoding="UTF-8"?>
<annotation>
  <size><width>640</width><height>480</height></size>
  <object>
    <name>UNKNOWN_CLASS</name>
    <bndbox>
      <xmin>10</xmin><ymin>10</ymin><xmax>100</xmax><ymax>100</ymax>
    </bndbox>
  </object>
</annotation>
"""
    p = tmp_path / "unknown.xml"
    p.write_text(xml_content, encoding="utf-8")
    return p


# ============================================================
# Fixture: mock ModelLoader
# ============================================================
@pytest.fixture
def mock_loader_loaded() -> MagicMock:
    """Mock ModelLoader yang sudah dimuat dengan kelas kerusakan jalan."""
    from src.model_loader import ModelStatus

    loader = MagicMock()
    status = ModelStatus(
        loaded=True,
        weights_path="models/road_damage_best.pt",
        model_type="road_damage",
        device="cpu",
        class_names=["D00", "D01", "D10", "D20", "D40", "D44"],
        num_classes=6,
    )
    loader.status = status
    loader.is_loaded.return_value = True

    # Mock model.predict -> kembalikan satu hasil dengan satu box
    mock_box = MagicMock()
    mock_box.xyxy.cpu.return_value.numpy.return_value = np.array([[50.0, 60.0, 200.0, 150.0]])
    mock_box.conf.cpu.return_value.numpy.return_value = np.array([0.82])
    mock_box.cls.cpu.return_value.numpy.return_value = np.array([0])

    mock_result = MagicMock()
    mock_result.boxes = mock_box

    loader.model.predict.return_value = [mock_result]
    return loader


@pytest.fixture
def mock_loader_not_loaded() -> MagicMock:
    """Mock ModelLoader yang belum dimuat."""
    from src.model_loader import ModelStatus

    loader = MagicMock()
    loader.status = ModelStatus(loaded=False)
    loader.is_loaded.return_value = False
    return loader


@pytest.fixture
def mock_loader_general() -> MagicMock:
    """Mock ModelLoader yang dimuat dengan model deteksi objek umum (COCO)."""
    from src.model_loader import ModelStatus

    loader = MagicMock()
    status = ModelStatus(
        loaded=True,
        weights_path="yolov8n.pt",
        model_type="general",
        device="cpu",
        class_names=["person", "car", "truck"],
        num_classes=80,
        warnings=["Model ini adalah model deteksi objek umum (COCO)."],
    )
    loader.status = status
    loader.is_loaded.return_value = True
    return loader

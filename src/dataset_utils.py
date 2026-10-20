"""
Utilitas dataset untuk RoadDamage AI — Bengkulu.

Menangani konversi format anotasi (RDD XML -> YOLO txt), validasi
struktur dataset, pemetaan kelas, serta pembuatan file data.yaml untuk
pelatihan dan evaluasi YOLO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from src.utils import LOGGER
from config import settings


# ============================================================
# Pemetaan kelas RDD (id string -> nama deskriptif)
# ============================================================
def get_rdd_class_names() -> Dict[str, str]:
    """Mengembalikan pemetaan id kelas RDD -> nama deskriptif."""
    return dict(settings.RDD_CLASSES)


def get_default_class_list() -> List[str]:
    """Mengembalikan daftar nama kelas YOLO (urutan indeks)."""
    return list(settings.DEFAULT_ROAD_DAMAGE_CLASSES)


# ============================================================
# Konversi anotasi RDD (XML) -> YOLO (txt)
# ============================================================
def rdd_xml_to_yolo(
    xml_path: Union[str, Path],
    image_width: int,
    image_height: int,
    class_id_map: Optional[Dict[str, int]] = None,
) -> List[str]:
    """
    Mengkonversi satu file anotasi RDD (format XML) ke baris YOLO.

    Format YOLO per baris: <class_id> <cx> <cy> <w> <h> (normalisasi).

    Catatan: format RDD asli menyimpan kotak sebagai xywh dalam piksel
    pada elemen <box>. Fungsi ini menangani struktur umum yang dipakai
    sekilab/RoadDamageDetector.
    """
    xml_path = Path(xml_path)
    if class_id_map is None:
        class_id_map = {c: i for i, c in enumerate(get_default_class_list())}

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Gagal membaca XML {xml_path}: {e}")

    lines: List[str] = []
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Lebar/tinggi citra harus > 0 untuk normalisasi.")

    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None or not name_el.text:
            continue
        cls_str = name_el.text.strip()
        if cls_str not in class_id_map:
            LOGGER.warning("Kelas '%s' tidak ada di pemetaan; dilewati.", cls_str)
            continue
        cls_id = class_id_map[cls_str]

        bnd = obj.find("bndbox")
        if bnd is None:
            # Beberapa RDD menggunakan <box> dengan atribut
            box_el = obj.find("box")
            if box_el is None:
                continue
            try:
                x = float(box_el.get("x"))
                y = float(box_el.get("y"))
                w = float(box_el.get("width"))
                h = float(box_el.get("height"))
            except (TypeError, ValueError):
                continue
            x1, y1 = x, y
            x2, y2 = x + w, y + h
        else:
            try:
                x1 = float(bnd.findtext("xmin", "0"))
                y1 = float(bnd.findtext("ymin", "0"))
                x2 = float(bnd.findtext("xmax", "0"))
                y2 = float(bnd.findtext("ymax", "0"))
            except (TypeError, ValueError):
                continue

        cx = (x1 + x2) / 2.0 / image_width
        cy = (y1 + y2) / 2.0 / image_height
        w = (x2 - x1) / image_width
        h = (y2 - y1) / image_height

        # Jepit ke [0, 1]
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        if w <= 0 or h <= 0:
            continue

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def convert_rdd_dataset(
    annotation_dir: Union[str, Path],
    image_dir: Union[str, Path],
    output_label_dir: Union[str, Path],
    class_id_map: Optional[Dict[str, int]] = None,
    image_exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> Tuple[int, int, List[str]]:
    """
    Mengkonversi seluruh dataset RDD (XML) di folder ke format YOLO (txt).

    Mengembalikan (jumlah_berhasil, jumlah_gagal, daftar_error).
    """
    annotation_dir = Path(annotation_dir)
    image_dir = Path(image_dir)
    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    if not annotation_dir.exists():
        return 0, 0, [f"Folder anotasi tidak ada: {annotation_dir}"]

    if class_id_map is None:
        class_id_map = {c: i for i, c in enumerate(get_default_class_list())}

    # Kumpulkan file XML
    xml_files = sorted(annotation_dir.glob("*.xml"))
    success = 0
    fail = 0
    errors: List[str] = []

    try:
        import xml.etree.ElementTree as ET
    except Exception:
        return 0, len(xml_files), ["Modul xml.etree tidak tersedia."]

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            fail += 1
            errors.append(f"{xml_path.name}: {e}")
            continue

        # Cari nama citra (elemen <filename> atau nama stem)
        filename_el = root.findtext("filename")
        size_el = root.find("size")
        img_w = 0
        img_h = 0
        if size_el is not None:
            img_w = int(size_el.findtext("width", "0") or "0")
            img_h = int(size_el.findtext("height", "0") or "0")

        stem = xml_path.stem
        # Jika ukuran tidak ada di XML, coba dapatkan dari citra
        if img_w <= 0 or img_h <= 0:
            img_path = None
            if filename_el:
                cand = image_dir / filename_el
                if cand.exists():
                    img_path = cand
            if img_path is None:
                for ext in image_exts:
                    cand = image_dir / f"{stem}{ext}"
                    if cand.exists():
                        img_path = cand
                        break
            if img_path is not None:
                try:
                    from PIL import Image
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                except Exception:
                    pass

        if img_w <= 0 or img_h <= 0:
            fail += 1
            errors.append(f"{xml_path.name}: ukuran citra tidak diketahui")
            continue

        lines = rdd_xml_to_yolo(xml_path, img_w, img_h, class_id_map)
        out_file = output_label_dir / f"{stem}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        success += 1

    LOGGER.info("Konversi RDD->YOLO: %d berhasil, %d gagal", success, fail)
    return success, fail, errors


# ============================================================
# Validasi struktur dataset YOLO
# ============================================================
@dataclass
class DatasetReport:
    """Laporan status dataset."""

    found: bool = False
    split_counts: Dict[str, int] = field(default_factory=dict)
    images_per_split: Dict[str, int] = field(default_factory=dict)
    labels_per_split: Dict[str, int] = field(default_factory=dict)
    missing_labels: List[str] = field(default_factory=list)
    empty_label_files: int = 0
    invalid_label_lines: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    data_yaml_path: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def total_images(self) -> int:
        return sum(self.images_per_split.values())

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "split_counts": self.split_counts,
            "images_per_split": self.images_per_split,
            "labels_per_split": self.labels_per_split,
            "total_images": self.total_images,
            "missing_labels": self.missing_labels[:20],
            "empty_label_files": self.empty_label_files,
            "invalid_label_lines": self.invalid_label_lines,
            "class_distribution": self.class_distribution,
            "data_yaml_path": self.data_yaml_path,
            "errors": self.errors,
        }


def validate_yolo_dataset(
    data_root: Union[str, Path] = None,
    image_subdirs: Sequence[str] = ("train", "val", "test"),
) -> DatasetReport:
    """
    Memvalidasi struktur dataset format YOLO.

    Asumsi struktur:
        data/images/<split>/*.jpg
        data/labels/<split>/*.txt
    """
    if data_root is None:
        data_root = settings.DATA_DIR
    data_root = Path(data_root)

    report = DatasetReport()

    images_root = data_root / "images"
    labels_root = data_root / "labels"

    if not images_root.exists() and not labels_root.exists():
        report.errors.append(
            f"Struktur dataset YOLO tidak ditemukan di {data_root}. "
            "Buat folder 'images/<split>' dan 'labels/<split>'. Lihat README."
        )
        return report

    report.found = True

    for split in image_subdirs:
        img_split = images_root / split
        lbl_split = labels_root / split
        if not img_split.exists():
            report.images_per_split[split] = 0
            continue

        img_files = []
        for ext in settings.SUPPORTED_IMAGE_EXTS:
            img_files.extend(img_split.glob(f"*{ext}"))
        report.images_per_split[split] = len(img_files)

        if not lbl_split.exists():
            report.errors.append(f"Folder label untuk split '{split}' tidak ada.")
            report.labels_per_split[split] = 0
            continue

        lbl_files = {p.stem: p for p in lbl_split.glob("*.txt")}
        report.labels_per_split[split] = len(lbl_files)

        for img in img_files:
            if img.stem not in lbl_files:
                report.missing_labels.append(img.name)
            else:
                lbl_path = lbl_files[img.stem]
                _scan_label_file(lbl_path, report)

    # Distribusi kelas
    report.split_counts = {
        k: report.images_per_split.get(k, 0) for k in image_subdirs
    }
    return report


# ============================================================
# Pembuatan data.yaml
# ============================================================
def write_data_yaml(
    data_root: Union[str, Path] = None,
    class_names: Sequence[str] = None,
    output_path: Union[str, Path] = None,
) -> Path:
    """
    Menulis file data.yaml untuk pelatihan/evaluasi YOLO.

    Struktur YOLO standar:
        path: <data_root>
        train: images/train
        val: images/val
        test: images/test
        names: {0: 'D00', ...}
    """
    if data_root is None:
        data_root = settings.DATA_DIR
    data_root = Path(data_root).resolve()

    if class_names is None:
        class_names = get_default_class_list()

    if output_path is None:
        output_path = data_root / "data.yaml"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    names_block = "\n".join(
        f"  {i}: {name}" for i, name in enumerate(class_names)
    )
    content = (
        f"# Dihasilkan oleh RoadDamage AI — Bengkulu\n"
        f"path: {data_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    LOGGER.info("data.yaml ditulis: %s", output_path)
    return output_path


def read_data_yaml(path: Union[str, Path]) -> Optional[dict]:
    """Membaca file data.yaml. Mengembalikan None jika gagal."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        try:
            import yaml  # type: ignore
        except ImportError:
            # Parser minimal manual
            return _parse_yaml_minimal(path)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        LOGGER.warning("Gagal membaca data.yaml: %s", e)
        return None


def _parse_yaml_minimal(path: Path) -> dict:
    """Parser YAML minimal untuk data.yaml tanpa dependensi PyYAML."""
    data: dict = {}
    names: dict = {}
    in_names = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "names:":
                in_names = True
                continue
            if in_names:
                if raw.startswith("  ") and ":" in stripped:
                    k, v = stripped.split(":", 1)
                    try:
                        names[int(k.strip())] = v.strip().strip("'\"")
                    except ValueError:
                        pass
                elif not raw.startswith(" "):
                    in_names = False
            if not in_names:
                if ":" in stripped:
                    k, v = stripped.split(":", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if v:
                        data[k] = v
    if names:
        data["names"] = names
    return data


# ============================================================
# Panduan dataset
# ============================================================
def get_dataset_instructions() -> str:
    """Mengembalikan teks panduan penyiapan dataset (untuk UI)."""
    return (
        "PANDUAN PENYIAPAN DATASET\n"
        "=========================\n\n"
        "1. Sumber utama: Road Damage Dataset (RDD) — github.com/sekilab/RoadDamageDetector\n"
        "   Unduh sesuai lisensi yang berlaku, lalu ekstrak.\n\n"
        "2. Struktur folder YOLO yang diharapkan:\n"
        "   data/\n"
        "   ├── images/\n"
        "   │   ├── train/  (*.jpg)\n"
        "   │   ├── val/    (*.jpg)\n"
        "   │   └── test/   (*.jpg)\n"
        "   ├── labels/\n"
        "   │   ├── train/  (*.txt, format YOLO)\n"
        "   │   ├── val/    (*.txt)\n"
        "   │   └── test/   (*.txt)\n"
        "   └── data.yaml\n\n"
        "3. Format YOLO per baris: <class_id> <cx> <cy> <w> <h> (normalisasi 0-1).\n\n"
        "4. Gunakan skrip konversi (src/dataset_utils.convert_rdd_dataset) jika\n"
        "   anotasi Anda masih dalam format XML RDD.\n\n"
        "5. Pisahkan data latih/validasi/uji untuk menghindari kebocoran data.\n\n"
        "6. JANGAN mengunggah dataset ke GitHub (sudah dikecualikan via .gitignore).\n"
    )


# Lampirkan helper scan sebagai fungsi modul agar dapat dipanggil
# oleh validate_yolo_dataset di atas.
def _scan_label_file(lbl_path: Path, report: DatasetReport) -> None:
    """Membaca file label YOLO dan memperbarui laporan."""
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            report.empty_label_files += 1
            return
        for line in content.splitlines():
            parts = line.split()
            if len(parts) != 5:
                report.invalid_label_lines += 1
                continue
            try:
                cls_id = int(parts[0])
                vals = [float(x) for x in parts[1:]]
            except ValueError:
                report.invalid_label_lines += 1
                continue
            if not all(0.0 <= v <= 1.0 for v in vals):
                report.invalid_label_lines += 1
                continue
            cls_key = f"class_{cls_id}"
            report.class_distribution[cls_key] = (
                report.class_distribution.get(cls_key, 0) + 1
            )
    except Exception as e:
        report.errors.append(f"{lbl_path.name}: {e}")

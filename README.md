# 🛣️ RoadDamage AI — Bengkulu

> **Evaluasi Kinerja Model YOLO untuk Deteksi Kerusakan Jalan  
> Menggunakan Citra Digital di Kota Bengkulu**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![License](https://img.shields.io/badge/Lisensi-MIT-green)

---

## Deskripsi

**RoadDamage AI** adalah aplikasi deteksi kerusakan jalan berbasis *computer vision*
yang dibangun menggunakan model YOLO (You Only Look Once) dan antarmuka web Streamlit.
Aplikasi ini dapat menerima citra dan video jalan, menampilkan lokasi kerusakan
menggunakan *bounding box*, serta mengevaluasi kinerja model menggunakan metrik
standar (*precision*, *recall*, mAP).

> ⚠️ **Disclaimer:** Hasil deteksi merupakan **bantuan identifikasi**, bukan pengganti
> inspeksi teknis jalan oleh tenaga berwenang.

---

## Latar Belakang Masalah

Identifikasi kerusakan jalan secara manual membutuhkan waktu dan tenaga yang tidak
sedikit. *Computer vision* dapat membantu mendeteksi berbagai jenis kerusakan —
seperti retak memanjang, retak melintang, retak buaya, dan lubang jalan — dari foto
jalan secara otomatis, sehingga proses pemeriksaan menjadi lebih cepat dan konsisten.

---

## Tujuan Penelitian

1. Mengimplementasikan model YOLO untuk mendeteksi kerusakan jalan dari citra digital.
2. Menampilkan lokasi kerusakan pada citra menggunakan *bounding box*.
3. Mengevaluasi kinerja model menggunakan *precision*, *recall*, dan mAP.
4. Menguji model pada citra jalan yang relevan dengan kondisi di Kota Bengkulu.
5. Membuat aplikasi yang dapat menerima gambar dan video.
6. Menyediakan hasil deteksi yang dapat diekspor dan didokumentasikan.
7. Menghasilkan proyek *computer vision* siap dipublikasikan ke GitHub.

---

## Fitur Aplikasi

| Fitur | Keterangan |
|-------|-----------|
| 🖼️ Deteksi Citra Tunggal | Upload JPG/PNG, deteksi kerusakan, lihat bounding box |
| 🗂️ Deteksi Batch | Upload banyak citra, rekap per-citra, ekspor CSV/JSON |
| 🎬 Deteksi Video | Proses video frame-demi-frame, simpan video beranotasi |
| 🧪 Evaluasi Model | Precision, Recall, mAP@0.5, mAP@0.5:0.95 (butuh dataset berlabel) |
| ⚙️ Pengaturan Model | Mode A (pralatih) & Mode B (kerusakan jalan), pilih CPU/GPU |
| 📊 Dashboard | Statistik sesi, distribusi kelas, ringkasan evaluasi |
| ⬇️ Ekspor Hasil | Unduh citra beranotasi (PNG), CSV, dan JSON |

---

## Teknologi

- **Python** 3.11
- **Ultralytics YOLO** (YOLOv8 / YOLO11)
- **PyTorch** (CPU mode dasar; GPU didukung bila CUDA tersedia)
- **OpenCV** — pemrosesan citra dan video
- **Streamlit** — antarmuka web
- **NumPy, Pandas, Pillow** — manipulasi data dan citra
- **Matplotlib, Plotly** — visualisasi
- **pytest** — pengujian otomatis

---

## Sumber Dataset

**Road Damage Dataset (RDD)**  
Repositori: https://github.com/sekilab/RoadDamageDetector  
Lisensi: CC BY 4.0 (periksa versi terkini)

Kelas kerusakan:

| Kode | Kerusakan |
|------|-----------|
| D00 | Retak memanjang — bekas roda |
| D01 | Retak memanjang — sambungan konstruksi |
| D10 | Retak melintang |
| D11 | Sambungan konstruksi melintang |
| D20 | Retak buaya (*alligator crack*) |
| D40 | Lepas butir / Lubang |
| D44 | Lubang jalan (*pothole*) |
| D50 | Garis putih pudar |
| D51 | Tanda lajur rusak |

---

## Struktur Folder

```
road-damage-ai-bengkulu/
├── app.py                  ← Entry point Streamlit (7 halaman)
├── requirements.txt
├── README.md
├── .gitignore
│
├── config/
│   ├── __init__.py
│   └── settings.py         ← Konfigurasi terpusat (jalur, kelas, parameter)
│
├── src/
│   ├── __init__.py
│   ├── model_loader.py     ← Pemuatan bobot YOLO + diagnostik
│   ├── image_detector.py   ← Inferensi citra tunggal
│   ├── video_detector.py   ← Inferensi video frame-demi-frame
│   ├── batch_processor.py  ← Pemrosesan batch citra
│   ├── dataset_utils.py    ← Konversi RDD → YOLO, validasi dataset
│   ├── evaluation.py       ← Evaluasi model (precision, recall, mAP)
│   ├── visualization.py    ← Gambar bounding box, grafik Plotly
│   └── utils.py            ← Validasi, ekspor CSV/JSON, utilitas umum
│
├── scripts/
│   ├── train.py            ← Pelatihan model (CLI)
│   ├── validate.py         ← Evaluasi model (CLI)
│   └── predict.py          ← Prediksi citra/folder (CLI)
│
├── tests/
│   ├── conftest.py         ← Fixtures bersama
│   ├── test_model_loader.py
│   ├── test_image_detector.py
│   └── test_dataset_utils.py
│
├── data/                   ← Dataset (tidak diunggah ke GitHub)
│   └── README.md
│
├── models/                 ← Bobot model (tidak diunggah ke GitHub)
│   └── README.md
│
└── outputs/                ← Hasil deteksi/evaluasi
    └── README.md
```

---

## Instalasi

### Prasyarat

- Python 3.11 (direkomendasikan)
- pip ≥ 24.0
- Windows / Linux / macOS
- (Opsional) NVIDIA GPU + CUDA ≥ 11.8 untuk akselerasi

### Langkah Instalasi

```powershell
# 1. Clone repositori
git clone https://github.com/<username>/road-damage-ai-bengkulu.git
cd road-damage-ai-bengkulu

# 2. Buat virtual environment
python -m venv .venv

# 3. Aktifkan virtual environment
.venv\Scripts\activate        # Windows PowerShell

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Pasang dependensi
pip install -r requirements.txt
```

### Dukungan GPU (Opsional)

Jika GPU NVIDIA tersedia, ganti baris PyTorch di `requirements.txt`:
```
# Hapus baris --extra-index-url CPU, ganti dengan:
--extra-index-url https://download.pytorch.org/whl/cu118
torch>=2.1.0,<2.6.0
torchvision>=0.16.0,<0.20.0
```
Kemudian di aplikasi, centang **Gunakan GPU bila tersedia** di halaman Pengaturan Model.

---

## Cara Menyiapkan Bobot Model

### Mode A — Bobot Pralatih Umum (uji coba saja)

Tidak diperlukan langkah tambahan. Pilih bobot di halaman **Pengaturan Model**;
Ultralytics akan mengunduhnya secara otomatis.

> ⚠️ Model ini hanya mendeteksi objek umum COCO — **bukan** kerusakan jalan.

### Mode B — Bobot Khusus Kerusakan Jalan (direkomendasikan)

1. Siapkan dataset (lihat bagian *Dataset* di bawah)
2. Latih model:
   ```powershell
   python -m scripts.train --data data/data.yaml --epochs 100 --batch 16 --imgsz 640 --device cpu
   ```
3. Salin bobot terbaik:
   ```powershell
   Copy-Item outputs\training\road_damage\weights\best.pt models\road_damage_best.pt
   ```
4. Muat model di halaman **Pengaturan Model** → Mode B.

---

## Cara Menjalankan Aplikasi

```powershell
# Aktifkan virtual environment terlebih dahulu
.venv\Scripts\activate

# Jalankan aplikasi
streamlit run app.py
```

Browser akan terbuka otomatis di `http://localhost:8501`.

---

## Cara Menyiapkan Dataset

```powershell
# 1. Unduh RDD dari github.com/sekilab/RoadDamageDetector
# 2. Konversi XML → YOLO
python -c "
from src.dataset_utils import convert_rdd_dataset
convert_rdd_dataset('path/ke/rdd/annotations', 'path/ke/rdd/images', 'data/labels/train')
"

# 3. Pindahkan citra ke data/images/<split>/

# 4. Buat data.yaml
python -c "from src.dataset_utils import write_data_yaml; write_data_yaml('data/')"
```

Atau gunakan antarmuka Streamlit: **Evaluasi Model** → *Panduan penyiapan dataset*.

---

## Cara Melakukan Prediksi (CLI)

```powershell
# Satu citra
python -m scripts.predict --weights models/road_damage_best.pt --input path/to/road.jpg --conf 0.25

# Folder (batch)
python -m scripts.predict --weights models/road_damage_best.pt --input path/to/folder --conf 0.25

# Tanpa menyimpan anotasi
python -m scripts.predict --weights models/road_damage_best.pt --input img.jpg --no-save-annotated
```

---

## Cara Melatih Model (CLI)

```powershell
# Pelatihan dasar (CPU)
python -m scripts.train --data data/data.yaml --epochs 100 --batch 16 --imgsz 640 --device cpu

# Lanjutkan pelatihan
python -m scripts.train --data data/data.yaml --epochs 150 --resume

# Gunakan GPU (jika tersedia)
python -m scripts.train --data data/data.yaml --epochs 100 --device 0
```

---

## Cara Mengevaluasi Model (CLI)

```powershell
# Evaluasi pada split test
python -m scripts.validate --weights models/road_damage_best.pt --data data/data.yaml --split test

# Simpan hasil sebagai JSON
python -m scripts.validate --weights models/road_damage_best.pt --data data/data.yaml --save-json
```

---

## Pengujian Otomatis

```powershell
# Jalankan semua pengujian
pytest tests/ -v

# Dengan cakupan kode
pytest tests/ -v --tb=short
```

Pengujian menggunakan mock dan fixture kecil — tidak memerlukan bobot YOLO atau
dataset nyata.

---

## Hasil Eksperimen

> Bagian ini akan diperbarui setelah bobot khusus kerusakan jalan dan dataset
> berlabel tersedia dan evaluasi nyata telah dijalankan.

Tanpa dataset dan bobot yang sesuai, nilai mAP, precision, dan recall tidak
tersedia dan tidak akan diarang oleh aplikasi ini.

---

## Keterbatasan

- Membutuhkan bobot yang dilatih khusus pada kelas kerusakan jalan untuk hasil yang relevan.
- Evaluasi model membutuhkan dataset berlabel (*ground truth*) yang terpisah dari data latih.
- Pemrosesan video dibatasi frame/durasi untuk efisiensi sumber daya.
- Performa pada kondisi pencahayaan rendah atau citra blur belum dioptimalkan.
- Hasil deteksi adalah bantuan identifikasi; keputusan perbaikan jalan memerlukan
  inspeksi lapangan.

---

## Referensi

- Jocher, G., et al. (2023). *Ultralytics YOLO*. https://github.com/ultralytics/ultralytics
- Maeda, H., et al. (2018). Road Damage Detection and Classification Using Deep Neural Networks with Smartphone Images. *Computer-Aided Civil and Infrastructure Engineering*.
- Arya, D., et al. (2022). RDD2022: A multi-national image dataset for automatic Road Damage Detection. *arXiv*. https://arxiv.org/abs/2209.08538
- Redmon, J., et al. (2016). You Only Look Once: Unified, Real-Time Object Detection. *CVPR*.

---

## Lisensi

Kode sumber berlisensi **MIT License**. Dataset (RDD) memiliki lisensi tersendiri
(CC BY 4.0 — periksa versi terkini di repositori RDD sebelum digunakan).

---

*Proyek ini dikembangkan sebagai bagian dari penelitian skripsi di bidang
computer vision untuk deteksi kerusakan jalan di Kota Bengkulu.*

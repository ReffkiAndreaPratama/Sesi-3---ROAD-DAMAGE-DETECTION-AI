# Folder `data/`

Folder ini menyimpan dataset untuk pelatihan, validasi, dan evaluasi model YOLO.

> ⚠️ **Dataset tidak diunggah ke GitHub** (sudah dikecualikan via `.gitignore`).  
> Ikuti panduan di bawah untuk mengunduh dan menempatkan dataset secara lokal.

---

## Struktur Folder yang Diharapkan

```
data/
├── images/
│   ├── train/    ← citra latih (*.jpg / *.jpeg / *.png)
│   ├── val/      ← citra validasi
│   └── test/     ← citra uji (tidak dipakai saat pelatihan)
├── labels/
│   ├── train/    ← anotasi format YOLO (*.txt)
│   ├── val/
│   └── test/
└── data.yaml     ← konfigurasi dataset YOLO (dibuat otomatis)
```

**Format anotasi YOLO per baris:**
```
<class_id> <cx> <cy> <w> <h>
```
Semua nilai `cx, cy, w, h` dinormalisasi ke `[0, 1]` terhadap dimensi citra.

---

## Sumber Dataset

### Road Damage Dataset (RDD) — Sumber Utama

| Info | Detail |
|------|--------|
| Repositori | https://github.com/sekilab/RoadDamageDetector |
| Lisensi | CC BY 4.0 (periksa lisensi terkini sebelum digunakan) |
| Format anotasi asli | XML (Pascal VOC) |
| Konversi ke YOLO | Gunakan `src/dataset_utils.convert_rdd_dataset()` |

### Kelas Kerusakan (RDD)

| ID | Nama |
|----|------|
| D00 | Retak memanjang (wheel mark) |
| D01 | Retak memanjang (sambungan konstruksi) |
| D10 | Retak melintang |
| D11 | Sambungan konstruksi |
| D20 | Retak buaya (alligator crack) |
| D40 | Lepas butir / Lubang |
| D44 | Lubang jalan (pothole) |
| D50 | Garis putih pudar |
| D51 | Tanda lajur rusak |

---

## Langkah Penyiapan Dataset

### 1 — Unduh dataset RDD
Kunjungi https://github.com/sekilab/RoadDamageDetector dan unduh dataset
sesuai lisensi yang berlaku. Ekstrak ke folder sementara.

### 2 — Konversi anotasi XML → YOLO
```python
from src.dataset_utils import convert_rdd_dataset

success, fail, errors = convert_rdd_dataset(
    annotation_dir="path/ke/annotations/",
    image_dir="path/ke/images/",
    output_label_dir="data/labels/train/",
)
print(f"Berhasil: {success}, Gagal: {fail}")
```

### 3 — Pindahkan citra ke folder yang sesuai
```
data/images/train/   ← citra latih
data/images/val/     ← citra validasi
data/images/test/    ← citra uji
```

### 4 — Buat data.yaml
```python
from src.dataset_utils import write_data_yaml
write_data_yaml("data/")
```

Atau dari antarmuka Streamlit: **Evaluasi Model** → *Buat data.yaml otomatis*.

### 5 — Pisahkan split dengan benar
- **Jangan** mencampur data latih dan data uji.
- Data uji tidak boleh digunakan selama pelatihan.
- Evaluasi resmi hanya boleh menggunakan split `test`.

---

## Pemisahan Data yang Benar

```
Total dataset  →  Train (70%)  |  Val (15%)  |  Test (15%)
```

Gunakan pemisahan acak tetap (`random_seed=42`, lihat `config/settings.py`).
Hindari kebocoran data — pastikan satu citra tidak muncul di dua split.

---

## Catatan

- JANGAN mengunggah dataset ke GitHub.
- JANGAN membuat anotasi palsu atau nilai mAP yang tidak diperoleh dari evaluasi nyata.
- Gunakan citra jalan asli di Kota Bengkulu sebagai data uji tambahan jika tersedia.

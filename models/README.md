# Folder `models/`

Folder ini menyimpan file bobot model YOLO (`.pt`).

> ⚠️ **File bobot tidak diunggah ke GitHub** (sudah dikecualikan via `.gitignore`).  
> Bobot harus diperoleh melalui pelatihan atau dari sumber yang sah.

---

## File Bobot yang Digunakan Aplikasi

| Nama File | Keterangan |
|-----------|-----------|
| `road_damage_best.pt` | Bobot utama — hasil pelatihan pada dataset kerusakan jalan. **Wajib ada** untuk deteksi yang relevan. |
| `road_damage_last.pt` | Checkpoint terakhir pelatihan (opsional). |

Nama default dikonfigurasi di `config/settings.py`:
```python
DEFAULT_ROAD_DAMAGE_WEIGHTS = MODELS_DIR / "road_damage_best.pt"
```

---

## Mode A — Bobot Pralatih Umum (untuk uji coba saja)

Model berikut dapat diunduh otomatis oleh Ultralytics, **tetapi tidak dilatih**
untuk mendeteksi kerusakan jalan (hanya mendeteksi objek umum COCO):

| Bobot | Keterangan |
|-------|-----------|
| `yolov8n.pt` | YOLOv8 Nano — paling ringan |
| `yolov8s.pt` | YOLOv8 Small |
| `yolov8m.pt` | YOLOv8 Medium |
| `yolov8l.pt` | YOLOv8 Large |
| `yolo11n.pt` | YOLO11 Nano |

> Gunakan Mode A hanya untuk **uji coba antarmuka**, bukan untuk menyimpulkan
> hasil deteksi kerusakan jalan.

---

## Mode B — Cara Mendapatkan Bobot Kerusakan Jalan

### Opsi 1: Latih dari awal
```bash
# Pastikan dataset sudah siap di data/
python -m scripts.train --data data/data.yaml --epochs 100 --batch 16 \
    --imgsz 640 --device cpu --weights yolov8s.pt --name road_damage
```

Setelah selesai, salin bobot terbaik:
```bash
# Windows PowerShell
Copy-Item outputs\training\road_damage\weights\best.pt models\road_damage_best.pt
```

### Opsi 2: Fine-tune dari bobot COCO
```bash
python -m scripts.train --data data/data.yaml --epochs 50 --batch 8 \
    --imgsz 640 --device cpu --weights yolov8s.pt
```

---

## Verifikasi Bobot

Setelah bobot ditempatkan, verifikasi melalui Streamlit:
1. Buka halaman **Pengaturan Model**
2. Pilih **Mode B — Bobot khusus kerusakan jalan**
3. Pilih file bobot
4. Klik **Muat Model**
5. Periksa status: model harus terdeteksi sebagai `road_damage`

---

## Catatan Penting

- JANGAN mengunggah file `.pt` ke GitHub (dapat melebihi 100 MB).
- Pastikan bobot benar-benar dilatih untuk kelas kerusakan jalan.
- Tanpa bobot yang sesuai, aplikasi dapat berjalan tetapi **tidak dapat**
  mendeteksi kerusakan jalan dengan benar.
- Nilai mAP, precision, dan recall hanya valid bila diperoleh dari evaluasi nyata
  menggunakan dataset berlabel yang terpisah dari data latih.

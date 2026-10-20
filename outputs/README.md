# Folder `outputs/`

Folder ini menyimpan hasil deteksi, evaluasi, dan pelatihan yang dihasilkan
oleh aplikasi. Subfolder dibuat otomatis saat pertama kali dibutuhkan.

> ⚠️ Isi folder ini tidak diunggah ke GitHub (dikecualikan via `.gitignore`).

---

## Subfolder

| Subfolder | Konten |
|-----------|--------|
| `outputs/detections/` | Citra beranotasi hasil deteksi tunggal |
| `outputs/evaluation/` | Hasil evaluasi model (JSON, grafik) |
| `outputs/videos/` | Video beranotasi hasil deteksi video |
| `outputs/training/` | Checkpoint dan log pelatihan |
| `outputs/uploads/` | File sementara citra yang diunggah |
| `outputs/uploads_batch/` | File sementara batch |
| `outputs/uploads_video/` | File sementara video |

---

## Format File Output

### Citra Beranotasi
- Format: `PNG`
- Penamaan: `<nama_asli>_<timestamp>.png`
- Berisi: bounding box, label kelas, confidence score

### Rekap Deteksi CSV
Satu baris per deteksi individual:
```
source, class_name, confidence, x1, y1, x2, y2, width, height, ...
```

### Rekap Deteksi JSON
```json
{
  "app": "RoadDamage AI — Bengkulu",
  "exported_at": "...",
  "total_images": 5,
  "total_detections": 12,
  "results": [...]
}
```

### Video Beranotasi
- Format: `MP4` (codec mp4v)
- Penamaan: `<nama_asli>_<timestamp>.mp4`

---

## Pembersihan Rutin

File sementara di `uploads/`, `uploads_batch/`, dan `uploads_video/` dapat
dihapus sewaktu-waktu. File deteksi dan evaluasi disimpan permanen hingga
dihapus secara manual.

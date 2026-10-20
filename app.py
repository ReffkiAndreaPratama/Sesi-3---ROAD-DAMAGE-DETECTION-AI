"""
RoadDamage AI — Bengkulu
========================
Aplikasi web deteksi kerusakan jalan berbasis YOLO.

Jalankan:
    streamlit run app.py

Antarmuka dalam Bahasa Indonesia. Hasil deteksi merupakan bantuan
identifikasi, bukan pengganti inspeksi teknis jalan.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Pastikan root proyek ada di sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.model_loader import (
    ModelLoader,
    ModelStatus,
    list_available_weights,
    get_pretrained_options,
    check_weights_exists,
)
from src.image_detector import ImageDetector
from src.batch_processor import BatchProcessor, validate_batch_files
from src.video_detector import VideoDetector
from src.evaluation import ModelEvaluator, get_evaluation_status_text
from src.dataset_utils import (
    validate_yolo_dataset,
    write_data_yaml,
    get_dataset_instructions,
    get_rdd_class_names,
    get_default_class_list,
)
from src.utils import (
    LOGGER,
    is_cuda_available,
    resolve_device,
    export_results_csv,
    export_results_json,
    validate_image_file,
    validate_video_file,
    safe_filename,
    timestamp_str,
    format_size,
)
from src.visualization import (
    annotate_image_rgb,
    class_distribution_plotly,
    confidence_histogram_plotly,
    confusion_matrix_plotly,
    results_to_bar_chart,
    image_to_png_bytes,
)


# ============================================================
# Inisialisasi state
# ============================================================
def init_state() -> None:
    defaults = {
        "loader": ModelLoader(),
        "model_loaded": False,
        "model_status": ModelStatus(),
        "images_processed": 0,
        "total_detections": 0,
        "class_distribution": {},
        "current_confidence": settings.DEFAULT_CONFIDENCE,
        "last_result": None,
        "last_batch": None,
        "last_video_summary": None,
        "last_evaluation": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    settings.ensure_directories()


# ============================================================
# Helper UI
# ============================================================
def get_loader() -> ModelLoader:
    return st.session_state.loader


def update_session_stats(result) -> None:
    """Memperbarui statistik sesi dari satu DetectionResult."""
    if result.is_error:
        return
    st.session_state.images_processed += 1
    st.session_state.total_detections += result.num_detections
    for cls, cnt in result.class_distribution().items():
        st.session_state.class_distribution[cls] = (
            st.session_state.class_distribution.get(cls, 0) + cnt
        )


def render_model_status_badge(status: ModelStatus, container=None):
    """Menampilkan badge status model."""
    c = container or st
    if status.loaded:
        type_label = {
            "road_damage": "Khusus Kerusakan Jalan",
            "general": "Deteksi Objek Umum",
            "unknown": "Tidak Diketahui",
        }.get(status.model_type, status.model_type)
        c.success(f"✅ Model dimuat — {type_label} ({status.num_classes} kelas) @ {status.device}")
    else:
        c.warning("⚠️ Model belum dimuat. Buka halaman **Pengaturan Model** untuk memuat bobot.")

    if status.warnings:
        for w in status.warnings:
            c.info(f"ℹ️ {w}")
    if status.errors:
        for e in status.errors:
            c.error(f"❌ {e}")


def section_header(title: str, icon: str = "") -> None:
    st.markdown(f"## {icon} {title}".strip())


# ============================================================
# CSS / Tema
# ============================================================
def set_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {max-width: 1200px; padding-top: 1.5rem;}
        .stMetric {background: #0e1117; border: 1px solid #262730; border-radius: 10px; padding: 12px 16px;}
        .badge {padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HALAMAN 1 — Dashboard
# ============================================================
def page_dashboard() -> None:
    section_header("Dashboard", "📊")
    render_model_status_badge(st.session_state.model_status)

    st.markdown("---")
    s = st.session_state
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gambar Diproses (sesi)", s.images_processed)
    col2.metric("Total Deteksi", s.total_detections)
    col3.metric("Confidence Threshold", f"{s.current_confidence:.2f}")
    col4.metric("Perangkat", s.model_status.device if s.model_status.loaded else "—")

    st.markdown("---")
    section_header("Distribusi Kelas Kerusakan (sesi)", "📈")
    if s.class_distribution:
        fig = class_distribution_plotly(
            s.class_distribution, "Distribusi Kelas — Sesi Berjalan"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Belum ada deteksi pada sesi ini. Jalankan deteksi di halaman Deteksi Citra atau Batch.")

    st.markdown("---")
    section_header("Ringkasan Evaluasi", "🧪")
    if s.last_evaluation is not None:
        ev = s.last_evaluation
        st.write(get_evaluation_status_text(ev))
        if ev.has_metrics:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Precision", f"{ev.precision:.3f}" if ev.precision is not None else "N/A")
            mc2.metric("Recall", f"{ev.recall:.3f}" if ev.recall is not None else "N/A")
            mc3.metric("mAP@0.5", f"{ev.map50:.3f}" if ev.map50 is not None else "N/A")
            mc4.metric("mAP@0.5:0.95", f"{ev.map50_95:.3f}" if ev.map50_95 is not None else "N/A")
            fig = results_to_bar_chart(
                {
                    "Precision": ev.precision or 0,
                    "Recall": ev.recall or 0,
                    "mAP@0.5": ev.map50 or 0,
                    "mAP@0.5:0.95": ev.map50_95 or 0,
                },
                title="Metrik Evaluasi",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Evaluasi belum dilakukan. Buka halaman **Evaluasi Model** setelah menyiapkan dataset berlabel.")

    st.markdown("---")
    st.caption(
        "⚠️ Hasil deteksi merupakan bantuan identifikasi, bukan pengganti inspeksi "
        "teknis jalan. Gunakan bobot YOLO yang dilatih khusus untuk kelas kerusakan jalan."
    )


# ============================================================
# HALAMAN 2 — Deteksi Citra
# ============================================================
def page_image_detection() -> None:
    section_header("Deteksi Citra", "🖼️")
    render_model_status_badge(st.session_state.model_status)

    if not st.session_state.model_loaded:
        st.stop()

    st.markdown("---")
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("### Pengaturan")
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.0, max_value=1.0,
            value=settings.DEFAULT_CONFIDENCE, step=0.05,
            help="Hanya tampilkan deteksi dengan keyakinan di atas ambang ini.",
        )
        img_size = st.select_slider(
            "Ukuran Input Model (px)",
            options=[320, 416, 512, 640, 800, 1024],
            value=settings.DEFAULT_IMAGE_SIZE,
        )
        save_annotated = st.checkbox("Simpan citra beranotasi", value=True)
        uploaded = st.file_uploader(
            "Unggah Citra Jalan",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        if uploaded is not None:
            st.caption(f"File: {uploaded.name} ({format_size(len(uploaded.getvalue()))})")

        run_btn = st.button("▶️ Jalankan Deteksi", type="primary", disabled=(uploaded is None))

    with col_r:
        st.markdown("### Hasil")
        if uploaded is None:
            st.info("Unggah citra JPG/JPEG/PNG untuk memulai deteksi.")
            st.stop()

        # Simpan unggahan ke file sementara
        tmp_dir = settings.OUTPUTS_DIR / "uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / safe_filename(uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())

        # Validasi
        ok, msg = validate_image_file(tmp_path)
        if not ok:
            st.error(f"File tidak valid: {msg}")
            st.stop()

        if run_btn:
            detector = ImageDetector(get_loader())
            with st.spinner("Menjalankan inferensi..."):
                result = detector.detect(
                    image_path=tmp_path,
                    confidence=confidence,
                    image_size=img_size,
                    save_annotated=save_annotated,
                )
            st.session_state.last_result = result
            st.session_state.current_confidence = confidence
            update_session_stats(result)

        result = st.session_state.last_result
        if result is None:
            st.info("Klik **Jalankan Deteksi** untuk memproses citra.")
            st.stop()

        if result.is_error:
            st.error(f"Gagal: {result.error}")
            st.stop()

        # Tampilkan sebelum/sesudah
        from src.utils import load_image_rgb
        orig = load_image_rgb(tmp_path)
        if orig is not None:
            annotated = annotate_image_rgb(orig, result.detections)
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Citra Asli")
                st.image(orig, use_container_width=True)
            with c2:
                st.caption("Hasil Deteksi")
                st.image(annotated, use_container_width=True)

            # Tombol unduh anotasi
            png_bytes = image_to_png_bytes(annotated)
            st.download_button(
                "⬇️ Unduh Citra Beranotasi (PNG)",
                data=png_bytes,
                file_name=f"annotated_{safe_filename(tmp_path.stem)}.png",
                mime="image/png",
            )

        # Tabel deteksi
        st.markdown("---")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Jumlah Deteksi", result.num_detections)
        mc2.metric("Waktu Inferensi (s)", f"{result.inference_time:.3f}")
        mc3.metric("Resolusi", f"{result.image_width}×{result.image_height}")

        if result.detections:
            st.markdown("#### Tabel Deteksi")
            table_rows = [
                {
                    "Kelas": d.class_name,
                    "Confidence": f"{d.confidence:.3f}",
                    "X1": f"{d.x1:.1f}", "Y1": f"{d.y1:.1f}",
                    "X2": f"{d.x2:.1f}", "Y2": f"{d.y2:.1f}",
                }
                for d in result.detections
            ]
            st.dataframe(table_rows, use_container_width=True)

            # Histogram confidence
            fig = confidence_histogram_plotly(result.detections)
            st.plotly_chart(fig, use_container_width=True)

            # Distribusi kelas
            dist = result.class_distribution()
            if dist:
                fig = class_distribution_plotly(dist, "Distribusi Kelas — Citra Ini")
                st.plotly_chart(fig, use_container_width=True)

            # Ekspor
            ec1, ec2 = st.columns(2)
            csv_path = settings.OUTPUTS_DIR / f"single_{timestamp_str()}.csv"
            export_results_csv([result], csv_path)
            with open(csv_path, "r", encoding="utf-8") as f:
                ec1.download_button("⬇️ Unduh CSV", f.read(), file_name=csv_path.name, mime="text/csv")
            json_path = settings.OUTPUTS_DIR / f"single_{timestamp_str()}.json"
            export_results_json([result], json_path)
            with open(json_path, "r", encoding="utf-8") as f:
                ec2.download_button("⬇️ Unduh JSON", f.read(), file_name=json_path.name, mime="application/json")
        else:
            st.info("Tidak ada kerusakan terdeteksi pada citra ini dengan ambang confidence saat ini.")


# ============================================================
# HALAMAN 3 — Deteksi Batch
# ============================================================
def page_batch_detection() -> None:
    section_header("Deteksi Batch", "🗂️")
    render_model_status_badge(st.session_state.model_status)

    if not st.session_state.model_loaded:
        st.stop()

    st.markdown("---")
    col_l, col_r = st.columns([1, 3])

    with col_l:
        st.markdown("### Pengaturan")
        confidence = st.slider(
            "Confidence Threshold", 0.0, 1.0,
            value=settings.DEFAULT_CONFIDENCE, step=0.05,
        )
        img_size = st.select_slider(
            "Ukuran Input (px)",
            options=[320, 416, 512, 640, 800, 1024],
            value=settings.DEFAULT_IMAGE_SIZE,
        )
        uploaded = st.file_uploader(
            "Unggah Beberapa Citra",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        st.caption(f"Maksimum {settings.MAX_BATCH_IMAGES} citra per batch.")
        run_btn = st.button("▶️ Proses Batch", type="primary", disabled=(not uploaded))

    with col_r:
        if not uploaded:
            st.info("Unggah beberapa citra untuk pemrosesan batch.")
            st.stop()

        if run_btn:
            # Simpan unggahan
            tmp_dir = settings.OUTPUTS_DIR / "uploads_batch"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for up in uploaded[: settings.MAX_BATCH_IMAGES]:
                p = tmp_dir / safe_filename(up.name)
                with open(p, "wb") as f:
                    f.write(up.getvalue())
                paths.append(p)

            valid, errs = validate_batch_files(paths)
            if errs:
                with st.expander(f"{len(errs)} file dilewati", expanded=False):
                    for e in errs:
                        st.write(f"• {e}")

            if not valid:
                st.error("Tidak ada citra valid untuk diproses.")
                st.stop()

            detector = ImageDetector(get_loader())
            batch = BatchProcessor(detector)

            progress_bar = st.progress(0.0, text="Memproses batch...")
            status_text = st.empty()

            def on_progress(p):
                pct = p.percent
                progress_bar.progress(min(pct / 100.0, 1.0),
                                       text=f"Memproses {p.done}/{p.total} — {p.current_file}")

            results, prog = batch.process(
                valid,
                confidence=confidence,
                image_size=img_size,
                save_annotated=False,
                callback=on_progress,
            )
            progress_bar.progress(1.0, text=f"Selesai: {prog.done}/{prog.total} ({prog.errors} error)")
            st.session_state.last_batch = batch
            st.session_state.current_confidence = confidence

            # Update statistik sesi
            for r in results:
                update_session_stats(r)

        batch = st.session_state.last_batch
        if batch is None or not batch.results:
            st.info("Klik **Proses Batch** untuk menjalankan deteksi pada citra yang diunggah.")
            st.stop()

        # Tabel ringkasan
        st.markdown("### Ringkasan Per Citra")
        st.dataframe(batch.summary_table(), use_container_width=True)

        # Metrik
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Citra", len(batch.results))
        b2.metric("Sukses", sum(1 for r in batch.results if not r.is_error))
        b3.metric("Error", sum(1 for r in batch.results if r.is_error))
        b4.metric("Total Deteksi", sum(r.num_detections for r in batch.results if not r.is_error))

        # Distribusi kelas agregat
        agg = batch.aggregate_distribution()
        if agg:
            st.markdown("### Distribusi Kelas (Agregat)")
            fig = class_distribution_plotly(agg, "Distribusi Kelas — Batch")
            st.plotly_chart(fig, use_container_width=True)

        # Ekspor
        st.markdown("### Ekspor Hasil")
        ec1, ec2 = st.columns(2)
        csv_path = batch.export_csv()
        with open(csv_path, "r", encoding="utf-8") as f:
            ec1.download_button("⬇️ Unduh Rekap CSV", f.read(), file_name=csv_path.name, mime="text/csv")
        json_path = batch.export_json()
        with open(json_path, "r", encoding="utf-8") as f:
            ec2.download_button("⬇️ Unduh Rekap JSON", f.read(), file_name=json_path.name, mime="application/json")


# ============================================================
# HALAMAN 4 — Deteksi Video
# ============================================================
def page_video_detection() -> None:
    section_header("Deteksi Video", "🎬")
    render_model_status_badge(st.session_state.model_status)

    if not st.session_state.model_loaded:
        st.stop()

    st.markdown("---")
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("### Pengaturan")
        confidence = st.slider(
            "Confidence Threshold", 0.0, 1.0,
            value=settings.DEFAULT_CONFIDENCE, step=0.05,
            key="vid_conf",
        )
        img_size = st.select_slider(
            "Ukuran Input (px)",
            options=[320, 416, 512, 640, 800, 1024],
            value=settings.DEFAULT_IMAGE_SIZE,
            key="vid_size",
        )
        max_frames = st.number_input(
            "Batas Frame", min_value=10, max_value=5000,
            value=settings.MAX_VIDEO_FRAMES, step=50,
            help="Batas aman untuk mencegah penggunaan sumber daya berlebihan.",
        )
        max_duration = st.number_input(
            "Batas Durasi (detik)", min_value=5, max_value=600,
            value=settings.MAX_VIDEO_DURATION_SECONDS, step=10,
        )
        save_output = st.checkbox("Simpan video hasil", value=True)
        uploaded = st.file_uploader(
            "Unggah Video",
            type=["mp4", "avi", "mov", "mkv"],
            accept_multiple_files=False,
        )
        if uploaded is not None:
            st.caption(f"File: {uploaded.name} ({format_size(len(uploaded.getvalue()))})")
        run_btn = st.button("▶️ Proses Video", type="primary", disabled=(uploaded is None))

    with col_r:
        if uploaded is None:
            st.info("Unggah video (MP4/AVI/MOV/MKV) untuk diproses. "
                    f"Ekstensi didukung: {', '.join(settings.SUPPORTED_VIDEO_EXTS)}")
            st.stop()

        # Simpan unggahan
        tmp_dir = settings.OUTPUTS_DIR / "uploads_video"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / safe_filename(uploaded.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getvalue())

        ok, msg = validate_video_file(tmp_path)
        if not ok:
            st.error(f"File tidak valid: {msg}")
            st.stop()

        if run_btn:
            vd = VideoDetector(get_loader())
            progress_bar = st.progress(0.0, text="Memproses video...")

            def on_progress(done, total):
                progress_bar.progress(min(done / max(total, 1), 1.0),
                                      text=f"Frame {done}/{total}")

            with st.spinner("Memproses frame..."):
                summary = vd.detect(
                    video_path=tmp_path,
                    confidence=confidence,
                    image_size=img_size,
                    max_frames=int(max_frames),
                    max_duration=int(max_duration),
                    save_output=save_output,
                    progress_callback=on_progress,
                )
            progress_bar.progress(1.0, text="Selesai")
            st.session_state.last_video_summary = summary

            if summary.class_distribution:
                for cls, cnt in summary.class_distribution.items():
                    st.session_state.class_distribution[cls] = (
                        st.session_state.class_distribution.get(cls, 0) + cnt
                    )

        summary = st.session_state.last_video_summary
        if summary is None:
            st.info("Klik **Proses Video** untuk memulai.")
            st.stop()

        if summary.is_error:
            st.error(f"Gagal: {summary.error}")
            st.stop()

        # Ringkasan
        v1, v2, v3, v4, v5 = st.columns(5)
        v1.metric("Frame Diproses", summary.processed_frames)
        v2.metric("Frame Dilewati", summary.skipped_frames)
        v3.metric("Total Deteksi", summary.total_detections)
        v4.metric("FPS Asli", f"{summary.fps:.1f}")
        v5.metric("Durasi Proses (s)", f"{summary.duration_seconds:.1f}")

        if summary.truncated:
            st.warning(f"Video dipotong pada batas frame/durasi untuk efisiensi sumber daya.")

        if summary.class_distribution:
            st.markdown("### Distribusi Kelas")
            fig = class_distribution_plotly(summary.class_distribution, "Distribusi Kelas — Video")
            st.plotly_chart(fig, use_container_width=True)

        if summary.output_path and Path(summary.output_path).exists():
            st.markdown("### Unduh Video Hasil")
            with open(summary.output_path, "rb") as f:
                st.download_button(
                    "⬇️ Unduh Video Beranotasi",
                    f.read(),
                    file_name=Path(summary.output_path).name,
                    mime="video/mp4",
                )
        elif summary.processed_frames > 0:
            st.info("Video hasil tersimpan di folder outputs/videos/.")


# ============================================================
# HALAMAN 5 — Evaluasi Model
# ============================================================
def page_evaluation() -> None:
    section_header("Evaluasi Model", "🧪")
    render_model_status_badge(st.session_state.model_status)

    st.markdown("---")

    # Status dataset
    st.markdown("### Status Dataset")
    report = validate_yolo_dataset(settings.DATA_DIR)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Citra Train", report.images_per_split.get("train", 0))
    c2.metric("Citra Val", report.images_per_split.get("val", 0))
    c3.metric("Citra Test", report.images_per_split.get("test", 0))
    c4.metric("Total", report.total_images)

    if report.errors:
        with st.expander("Catatan dataset", expanded=True):
            for e in report.errors:
                st.write(f"• {e}")

    if not report.found or report.total_images == 0:
        st.warning(
            "Dataset berlabel belum tersedia. Evaluasi membutuhkan ground truth "
            "(anotasi) pada split yang terpisah dari data latih. Tanpa dataset, "
            "**metrik tidak akan dihitung** dan tidak ada nilai yang diarang."
        )
        with st.expander("📖 Panduan penyiapan dataset"):
            st.code(get_dataset_instructions())
        st.stop()

    # Tombol buat data.yaml
    if not (settings.DATA_DIR / "data.yaml").exists():
        if st.button("📄 Buat data.yaml otomatis"):
            p = write_data_yaml(settings.DATA_DIR)
            st.success(f"data.yaml dibuat: {p}")
            st.rerun()

    st.markdown("---")
    st.markdown("### Jalankan Evaluasi")

    if not st.session_state.model_loaded:
        st.warning("Muat model terlebih dahulu di halaman Pengaturan Model.")
        st.stop()

    col_ev, col_opt = st.columns([1, 2])
    with col_opt:
        split = st.selectbox("Split Evaluasi", ["test", "val"], index=0)
        img_size = st.select_slider("Ukuran Input (px)",
                                    options=[320, 416, 512, 640, 800, 1024],
                                    value=settings.DEFAULT_IMAGE_SIZE)
        batch = st.number_input("Batch Evaluasi", min_value=1, max_value=32, value=1)
    with col_ev:
        run_btn = st.button("▶️ Jalankan Evaluasi", type="primary")

    if run_btn:
        data_yaml = settings.DATA_DIR / "data.yaml"
        if not data_yaml.exists():
            write_data_yaml(settings.DATA_DIR)

        evaluator = ModelEvaluator(get_loader())
        with st.spinner("Menjalankan validasi... (mungkin butuh waktu)"):
            metrics = evaluator.evaluate(
                data_yaml=data_yaml,
                image_size=img_size,
                batch=int(batch),
                split=split,
                save_dir=settings.OUTPUTS_EVAL_DIR / "run",
            )
        st.session_state.last_evaluation = metrics

    metrics = st.session_state.last_evaluation
    if metrics is None:
        st.info("Klik **Jalankan Evaluasi** untuk menghitung metrik pada dataset berlabel.")
        st.stop()

    st.markdown("---")
    st.markdown("### Hasil Evaluasi")
    st.write(get_evaluation_status_text(metrics))

    if metrics.errors and not metrics.has_metrics:
        for e in metrics.errors:
            st.error(f"• {e}")

    if metrics.has_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{metrics.precision:.4f}" if metrics.precision is not None else "N/A")
        m2.metric("Recall", f"{metrics.recall:.4f}" if metrics.recall is not None else "N/A")
        m3.metric("mAP@0.5", f"{metrics.map50:.4f}" if metrics.map50 is not None else "N/A")
        m4.metric("mAP@0.5:0.95", f"{metrics.map50_95:.4f}" if metrics.map50_95 is not None else "N/A")

        fig = results_to_bar_chart(
            {
                "Precision": metrics.precision or 0,
                "Recall": metrics.recall or 0,
                "mAP@0.5": metrics.map50 or 0,
                "mAP@0.5:0.95": metrics.map50_95 or 0,
            },
            title="Metrik Utama",
        )
        st.plotly_chart(fig, use_container_width=True)

        if metrics.inference_speed_ms is not None:
            st.metric("Kecepatan Inferensi (ms/gambar)", f"{metrics.inference_speed_ms:.2f}")

        if metrics.per_class:
            st.markdown("#### Metrik per Kelas")
            st.dataframe(metrics.per_class, use_container_width=True)

        if metrics.confusion_matrix and metrics.confusion_matrix_classes:
            try:
                cm = np.array(metrics.confusion_matrix)
                fig = confusion_matrix_plotly(cm, metrics.confusion_matrix_classes)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass


# ============================================================
# HALAMAN 6 — Pengaturan Model
# ============================================================
def page_model_settings() -> None:
    section_header("Pengaturan Model", "⚙️")
    render_model_status_badge(st.session_state.model_status)

    st.markdown("---")

    # Info perangkat
    gpu_ok = is_cuda_available()
    st.markdown("### Perangkat")
    st.write(f"CPU tersedia. CUDA: {'✅ terdeteksi' if gpu_ok else '❌ tidak tersedia'} "
             f"(mode dasar: CPU)")

    st.markdown("---")
    st.markdown("### Pilihan Mode Model")

    mode = st.radio(
        "Mode",
        ["A — Bobot pralatih umum (peringatan)", "B — Bobot khusus kerusakan jalan"],
        index=1,
        help="Mode A memuat model deteksi objek umum (bukan kerusakan jalan). "
             "Mode B memuat bobot yang Anda latih untuk kelas kerusakan jalan.",
    )

    prefer_gpu = st.checkbox("Gunakan GPU bila tersedia", value=False)
    if prefer_gpu and not gpu_ok:
        st.warning("GPU tidak terdeteksi; akan tetap menggunakan CPU.")

    weights_to_load = None

    if mode.startswith("A"):
        st.markdown("#### Bobot Pralatih Umum")
        st.warning(
            "⚠️ Bobot ini dilatih pada dataset COCO (objek umum), BUKAN kerusakan "
            "jalan. Hasil deteksi hanya mendeteksi objek umum. Gunakan hanya untuk "
            "uji coba antarmuka, bukan untuk menyimpulkan kerusakan jalan."
        )
        opts = get_pretrained_options()
        choice = st.selectbox("Pilih bobot", list(opts.keys()),
                              format_func=lambda x: f"{x} — {opts[x]}")
        weights_to_load = choice
    else:
        st.markdown("#### Bobot Khusus Kerusakan Jalan")
        available = list_available_weights(settings.MODELS_DIR)
        default_w = settings.DEFAULT_ROAD_DAMAGE_WEIGHTS.name

        if available:
            choice = st.selectbox(
                "Bobot tersedia di models/",
                available,
                index=available.index(default_w) if default_w in available else 0,
            )
            weights_to_load = str(settings.MODELS_DIR / choice)
        else:
            st.info(
                f"Belum ada bobot di folder models/. Tempatkan file .pt hasil "
                f"pelatihan di `{settings.MODELS_DIR}` (mis. "
                f"`road_damage_best.pt`)."
            )
            custom = st.text_input(
                "atau jalur bobot kustom (.pt)",
                value=str(settings.DEFAULT_ROAD_DAMAGE_WEIGHTS),
            )
            weights_to_load = custom

    # Cek eksistensi
    exists, exists_msg = check_weights_exists(weights_to_load)
    if exists:
        st.caption(f"✅ {exists_msg}")
    else:
        st.caption(f"❌ {exists_msg}")

    load_btn = st.button("🔄 Muat Model", type="primary")

    if load_btn:
        with st.spinner("Memuat bobot..."):
            status = get_loader().load(weights_to_load, prefer_gpu=prefer_gpu)
        st.session_state.model_status = status
        st.session_state.model_loaded = status.loaded
        if status.loaded:
            st.success(f"Model berhasil dimuat ({status.model_type}, {status.num_classes} kelas) @ {status.device}")
        else:
            st.error("Gagal memuat model. Lihat detail error di bawah.")
        render_model_status_badge(status)
        st.rerun()

    # Panduan pelatihan
    st.markdown("---")
    with st.expander("📖 Cara mendapatkan bobot khusus kerusakan jalan"):
        st.markdown(
            """
            1. Siapkan dataset format YOLO (lihat halaman **Evaluasi Model** → panduan dataset).
            2. Jalankan pelatihan:
               ```
               python -m scripts.train --data data/data.yaml --epochs 100 --batch 16 --imgsz 640 --device cpu
               ```
            3. Salin bobot terbaik ke `models/`:
               ```
               copy outputs\\training\\road_damage\\weights\\best.pt models\\road_damage_best.pt
               ```
            4. Kembali ke halaman ini, pilih Mode B, lalu **Muat Model**.

            Tanpa bobot dan dataset yang sesuai, aplikasi dapat berjalan tetapi
            belum tentu mendeteksi kerusakan jalan dengan benar.
            """
        )


# ============================================================
# HALAMAN 7 — Tentang Proyek
# ============================================================
def page_about() -> None:
    section_header("Tentang Proyek", "ℹ️")

    st.markdown(f"### {settings.APP_NAME}")
    st.markdown(f"**Judul Penelitian:** {settings.RESEARCH_TITLE}")
    st.markdown(settings.APP_DESCRIPTION)

    st.markdown("---")
    st.markdown("#### Latar Belakang")
    st.markdown(
        "Identifikasi kerusakan jalan secara manual membutuhkan waktu dan tenaga. "
        "Computer Vision dapat digunakan untuk membantu mendeteksi kerusakan dari "
        "foto jalan, sehingga hasilnya dapat diperiksa lebih cepat dan konsisten."
    )

    st.markdown("#### Tujuan")
    goals = [
        "Mengimplementasikan YOLO untuk deteksi kerusakan jalan dari citra digital.",
        "Menampilkan lokasi kerusakan pada citra menggunakan bounding box.",
        "Mengevaluasi kinerja model menggunakan precision, recall, dan mAP.",
        "Menguji model pada citra jalan yang relevan dengan kondisi di Kota Bengkulu.",
        "Membuat aplikasi yang dapat menerima gambar dan video.",
        "Menyediakan hasil deteksi yang dapat diekspor dan didokumentasikan.",
    ]
    for g in goals:
        st.markdown(f"- {g}")

    st.markdown("---")
    st.markdown("#### Teknologi")
    techs = [
        "Python 3.11", "Ultralytics YOLO", "PyTorch (CPU/GPU)", "OpenCV",
        "Streamlit", "NumPy & Pandas", "Pillow", "Matplotlib & Plotly", "pytest",
    ]
    st.markdown(" · ".join(techs))

    st.markdown("#### Kelas Kerusakan (Road Damage Dataset)")
    rdd = get_rdd_class_names()
    rows = [{"Kode": k, "Nama": v} for k, v in rdd.items()]
    st.dataframe(rows, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Sumber Dataset")
    st.markdown(
        "Road Damage Dataset (RDD): [github.com/sekilab/RoadDamageDetector]"
        "(https://github.com/sekilab/RoadDamageDetector)"
    )

    st.markdown("#### Keterbatasan")
    limits = [
        "Model membutuhkan bobot yang dilatih khusus pada kelas kerusakan jalan.",
        "Evaluasi membutuhkan dataset berlabel (ground truth) yang terpisah dari data latih.",
        "Hasil deteksi merupakan bantuan identifikasi, bukan pengganti inspeksi teknis jalan.",
        "Pemrosesan video dibatasi jumlah frame/durasi untuk efisiensi sumber daya.",
    ]
    for l in limits:
        st.markdown(f"- {l}")

    st.markdown("---")
    st.warning(
        "⚠️ **Disclaimer:** Hasil deteksi dari aplikasi ini adalah bantuan identifikasi "
        "berbasis computer vision, bukan pengganti inspeksi teknis jalan oleh tenaga "
        "berwenang. Keputusan terkait perbaikan jalan harus melalui pemeriksaan lapangan."
    )


# ============================================================
# Navigasi
# ============================================================
PAGES = {
    "📊 Dashboard": page_dashboard,
    "🖼️ Deteksi Citra": page_image_detection,
    "🗂️ Deteksi Batch": page_batch_detection,
    "🎬 Deteksi Video": page_video_detection,
    "🧪 Evaluasi Model": page_evaluation,
    "⚙️ Pengaturan Model": page_model_settings,
    "ℹ️ Tentang Proyek": page_about,
}


def main() -> None:
    st.set_page_config(
        page_title=settings.APP_TITLE,
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    set_custom_css()
    init_state()

    st.sidebar.markdown(f"## 🛣️ {settings.APP_NAME}")
    st.sidebar.caption(settings.APP_SUBTITLE)
    st.sidebar.markdown("---")

    selection = st.sidebar.radio("Navigasi", list(PAGES.keys()))
    st.sidebar.markdown("---")

    # Status singkat di sidebar
    status = st.session_state.model_status
    if status.loaded:
        st.sidebar.success(f"Model: {status.model_type} @ {status.device}")
    else:
        st.sidebar.warning("Model belum dimuat")

    st.sidebar.metric("Gambar Diproses", st.session_state.images_processed)
    st.sidebar.metric("Total Deteksi", st.session_state.total_detections)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"v{settings.APP_VERSION}")

    PAGES[selection]()


if __name__ == "__main__":
    main()

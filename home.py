import streamlit as st
import sqlite3
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

def girly_style():
    st.markdown("""
    <style>
    .block-container {
      background: linear-gradient(135deg, #ffe4e6, #f8bbd0) !important;
      border-radius: 15px;
      padding: 2rem;
    }
    .css-18e3th9 {
      background: linear-gradient(135deg, #ffe4e6, #f8bbd0) !important;
    }
    .reportview-container, .main, header, footer {
        background: linear-gradient(135deg, #ffe4e6, #f8bbd0);
        color: #880e4f;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ff80ab, #f48fb1);
        color: #4a148c;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    [data-testid="stSidebar"] h2 {
        color: #880e4f !important;
    }
    button[kind="primary"] {
        background-color: #ec407a !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: bold !important;
    }
    button[kind="primary"]:hover {
        background-color: #d81b60 !important;
        color: white !important;
    }
    .stCodeBlock pre {
        background-color: #fce4ec !important;
        color: #880e4f !important;
        font-family: 'Courier New', Courier, monospace !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    .stTable thead tr th {
        background-color: #f48fb1 !important;
        color: #4a148c !important;
    }
    .stInfo {
        background-color: #f8bbd0 !important;
        border-radius: 10px !important;
        color: #880e4f !important;
    }
    </style>
    """, unsafe_allow_html=True)

def sidebar_header():
    st.sidebar.markdown("### Selamat datang 👋")
    st.sidebar.markdown("---")
    st.sidebar.caption("👩‍💻 Dikembangkan oleh Chisilia Amanda Wahyudi untuk skripsi deteksi penyakit buah jambu biji 🍈")

def home_page():
    st.title("🍈 Guava Disease Detection!")
    st.write("Selamat datang di aplikasi YOLOv11 untuk deteksi penyakit jambu biji berbasis web.")
    st.markdown("""
    **Tentang Jambu Biji dan Penyakitnya**  
    Jambu biji (Psidium guajava) adalah buah tropis kaya vitamin C. Namun, buah ini rentan terhadap beberapa penyakit utama:

    1. **Phytophthora**: Busuk akar dan batang.
    2. **Scab**: Bercak kasar coklat di kulit.
    3. **Styler and Root**: Mengganggu pembentukan buah dan penyerapan nutrisi.
    """)

def detection_page():
    st.title("🔍 Deteksi Penyakit Jambu menggunakan YOLOv11")

    ROOT = Path(__file__).parent
    IMAGES_DIR = ROOT / 'images'
    MODEL_DIR = ROOT / 'weights'

    DEFAULT_IMAGE = IMAGES_DIR / 'jambu1.jpg'
    DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detectedimage1.png'
    DETECTION_MODEL = MODEL_DIR / 'best.pt'

    st.sidebar.header("🔧 Konfigurasi Model")
    confidence_value = float(st.sidebar.slider("Confidence", 10, 100, 25)) / 100

    try:
        model = YOLO(DETECTION_MODEL)
        st.sidebar.success("✅ Model berhasil dimuat")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

    st.sidebar.markdown("### 🏷️ Label yang dikenali:")
    for i, name in model.names.items():
        st.sidebar.write(f"{i}: {name}")

    source_radio = st.sidebar.radio("Sumber Gambar", ["Image", "Video", "Camera"])

    if source_radio == "Image":
        source_image = st.sidebar.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])
        col1, col2 = st.columns(2)

        with col1:
            if source_image:
                image = Image.open(source_image)
                st.image(image, caption="Gambar yang diunggah", use_container_width=True)
            else:
                if DEFAULT_IMAGE.exists():
                    image = Image.open(DEFAULT_IMAGE)
                    st.image(image, caption="Gambar default", use_container_width=True)
                else:
                    st.warning("❌ Gambar default tidak ditemukan.")
                    return  # agar tidak lanjut ke pemanggilan predict()

        with col2:
            if st.sidebar.button("🔍 Deteksi Objek"):
                result = model.predict(image, conf=confidence_value)
                boxes = result[0].boxes
                result_img = result[0].plot()[:, :, ::-1]

                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
                st.success(f"✅ Jumlah objek terdeteksi: {len(boxes)}")

                st.write("### Results (xyxy)")
                st.code(str(boxes.xyxy.cpu()), language="python")

                detections = []
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = model.names.get(cls_id, "Unknown")
                    conf = box.conf[0].item()
                    detections.append({"Label": label, "Confidence": f"{conf:.2f}"})
                st.table(detections)

    elif source_radio == "Video":
        video_file = st.sidebar.file_uploader("Unggah video...", type=["mp4", "mov", "avi"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result = model(frame[..., ::-1], conf=confidence_value)
                frame_output = result[0].plot()
                stframe.image(frame_output, channels="RGB")
            cap.release()

    elif source_radio == "Camera":
        run = st.checkbox("🔴 Jalankan Kamera")
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Kamera tidak terdeteksi.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    break
                result = model(frame[..., ::-1], conf=confidence_value)
                frame_output = result[0].plot()
                FRAME_WINDOW.image(frame_output, channels="RGB")
            cap.release()

def main():
    girly_style()
    sidebar_header()
    page = st.sidebar.radio("Menu", ["Home", "Detection"])
    if page == "Home":
        home_page()
    elif page == "Detection":
        detection_page()

if __name__ == "__main__":
    main()

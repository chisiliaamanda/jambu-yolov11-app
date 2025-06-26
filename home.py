import streamlit as st
import json
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
    .stTable thead tr th {
        background-color: #f48fb1 !important;
        color: #4a148c !important;
    }
    </style>
    """, unsafe_allow_html=True)

def sidebar_header():
    st.sidebar.markdown("### Selamat datang 👋")
    st.sidebar.markdown("---")
    st.sidebar.caption("👩‍💻 Oleh Chisilia Amanda Wahyudi | Skripsi Deteksi Penyakit Jambu 🍈")

def home_page():
    st.title("🍈 Guava Disease Detection with YOLOv11")
    st.markdown("""
    Deteksi penyakit jambu biji secara otomatis menggunakan model YOLOv11.

    **Jenis penyakit:**
    1. Phytophthora → busuk akar & batang  
    2. Scab → bercak kasar di kulit  
    3. Styler and Root → gangguan bunga & akar
    """)
    st.image("images/jambu1.jpg", caption="Contoh Jambu Biji", use_column_width=True)

def detection_page():
    st.title("🔍 Deteksi Penyakit Jambu")

    ROOT = Path(__file__).parent
    IMAGES_DIR = ROOT / 'images'
    MODEL_DIR = ROOT / 'weights'
    JSON_PATH = ROOT / 'penyakit_jambu_info.json'

    DEFAULT_IMAGE = IMAGES_DIR / 'jambu1.jpg'
    DEFAULT_RESULT = IMAGES_DIR / 'detectedimage1.png'
    DETECTION_MODEL = MODEL_DIR / 'best.pt'

    confidence = st.sidebar.slider("Confidence", 10, 100, 25) / 100
    input_mode = st.sidebar.radio("Sumber Gambar", ["Image", "Video", "Camera"])

    try:
        model = YOLO(DETECTION_MODEL)
        st.sidebar.success("✅ Model berhasil dimuat")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return

    st.sidebar.markdown("### 🏷️ Label:")
    for i, name in model.names.items():
        st.sidebar.write(f"{i}: {name}")

    if 'history' not in st.session_state:
        st.session_state.history = []

    def tampilkan_penjelasan(label_list):
        try:
            with open(JSON_PATH, "r", encoding="utf-8") as f:
                info = json.load(f)
            for label in label_list:
                st.info(f"**{label}**: {info.get(label, 'Info tidak tersedia')}")
        except:
            st.warning("🔍 File penjelasan tidak ditemukan")

    if input_mode == "Image":
        uploaded = st.sidebar.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
        col1, col2 = st.columns(2)

        with col1:
            if uploaded:
                img = Image.open(uploaded)
            elif DEFAULT_IMAGE.exists():
                img = Image.open(DEFAULT_IMAGE)
            else:
                st.warning("Tidak ada gambar.")
                return
            st.image(img, caption="Gambar Input", use_column_width=True)

        with col2:
            if st.sidebar.button("🔎 Deteksi Objek"):
                result = model.predict(img, conf=confidence)
                boxes = result[0].boxes
                hasil = result[0].plot()[:, :, ::-1]
                st.image(hasil, caption="Hasil Deteksi", use_column_width=True)

                detected_labels = []
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = model.names.get(cls_id, "Unknown")
                    detected_labels.append(label)

                tampilkan_penjelasan(set(detected_labels))

                st.session_state.history.append({
                    'type': 'Image',
                    'input_img': np.array(img.convert("RGB")).tolist(),
                    'result_img': hasil.tolist(),
                    'labels': detected_labels
                })

    elif input_mode == "Video":
        video = st.sidebar.file_uploader("Unggah Video", type=["mp4", "mov"])
        if video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result = model(frame[..., ::-1], conf=confidence)
                plotted = result[0].plot()
                stframe.image(plotted, channels="RGB")
            cap.release()

    elif input_mode == "Camera":
        camera_img = st.camera_input("📷 Ambil foto dari kamera")
        if camera_img:
            img = Image.open(camera_img)
            result = model.predict(img, conf=confidence)
            plotted = result[0].plot()[:, :, ::-1]
            st.image(plotted, caption="Hasil Deteksi", use_column_width=True)

            # Simpan ke history
            st.session_state.history.append({
                'type': 'Camera',
                'input_img': np.array(img.convert("RGB")).tolist(),
                'result_img': plotted.tolist(),
                'labels': [model.names[int(box.cls[0].item())] for box in result[0].boxes]
            })

def history_page():
    st.title("📜 Riwayat Deteksi")
    if 'history' not in st.session_state or not st.session_state.history:
        st.info("Belum ada deteksi yang disimpan.")
    else:
        for i, item in enumerate(reversed(st.session_state.history), 1):
            st.subheader(f"Riwayat #{i}")
            st.image(np.array(item['input_img'], dtype=np.uint8), caption="Input", use_column_width=True)
            st.image(np.array(item['result_img'], dtype=np.uint8), caption="Hasil Deteksi", use_column_width=True)
            st.markdown("**Penjelasan:**")
            for label in item['labels']:
                st.markdown(f"- {label}")

def main():
    girly_style()
    sidebar_header()
    menu = st.sidebar.radio("📌 Menu", ["Home", "Detection", "History"])
    if menu == "Home":
        home_page()
    elif menu == "Detection":
        detection_page()
    elif menu == "History":
        history_page()

if __name__ == "__main__":
    main()

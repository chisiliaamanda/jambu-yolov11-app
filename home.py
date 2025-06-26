import sqlite3
import hashlib
import PIL
import streamlit as st
from pathlib import Path
import json
import helper
import settings
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Deteksi Penyakit Jambu Biji menggunakan YOLOv11",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name, password FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user[0] if user and user[1] == hash_password(password) else None

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = True
    st.session_state['username'] = "admin"
    st.session_state['name'] = "Admin"

ENABLE_LOGIN = False

if ENABLE_LOGIN and st.session_state['authentication_status'] != True:
    st.header("Login 🍈")
    st.info("🔒 Username = admin, Password = 123")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        name = verify_user(username, password)
        if name:
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.session_state['name'] = name
            st.success("Berhasil login")
            st.experimental_rerun()
        else:
            st.error("Username atau password salah")
else:
    def main():
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False

        st.sidebar.title("Selamat Datang")
        st.sidebar.header("🍈 Deteksi Jambu Biji")

        menu = st.sidebar.radio("Menu", ["Home", "Detection", "Riwayat"])

        if menu == "Home":
            st.markdown("## 🍎 Deteksi Penyakit Jambu menggunakan YOLOv11")
            col1, col2 = st.columns(2)
            with col1:
                st.image("images/jambu1.jpg", caption="Gambar default", use_column_width=True)
            with col2:
                st.image("images/detectedimage1.png", caption="Deteksi Default", use_column_width=True)

        elif menu == "Detection":
            st.markdown("## 🍎 Deteksi Penyakit Jambu menggunakan YOLOv11")
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
                    image = Image.open(source_image) if source_image else Image.open(DEFAULT_IMAGE)
                    st.image(image, caption="Gambar default", use_column_width=True)

                with col2:
                    if st.sidebar.button("🔍 Deteksi Objek"):
                        result = model.predict(image, conf=confidence_value, imgsz=320)
                        boxes = result[0].boxes
                        result_img = result[0].plot()[:, :, ::-1]
                        st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

                        detections = []
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            label = model.names.get(cls_id, "Unknown")
                            conf = box.conf[0].item()
                            detections.append({"Label": label, "Confidence": f"{conf:.2f}"})
                        if detections:
                            st.table(detections)

        elif menu == "Riwayat":
            st.markdown("## 📜 Riwayat Deteksi")
            st.info("Belum ada riwayat deteksi disimpan dalam sesi ini.")

        st.sidebar.markdown("---")
        st.session_state.dark_mode = st.sidebar.checkbox("Dark Mode", value=st.session_state.dark_mode)
        st.sidebar.image("images/jambu1.jpg", use_column_width=True)

    if __name__ == "__main__" or True:
        main()

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
    /* Background pink soft untuk konten utama */
    .block-container {
      background: linear-gradient(135deg, #ffe4e6, #f8bbd0) !important;
      border-radius: 15px;
      padding: 2rem;
    }

    /* Background body page */
    .css-18e3th9 {
      background: linear-gradient(135deg, #ffe4e6, #f8bbd0) !important;
    }

    /* Background pink soft lama */
    .reportview-container, .main, header, footer {
        background: linear-gradient(135deg, #ffe4e6, #f8bbd0);
        color: #880e4f;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    /* Sidebar pink-orange gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #ff80ab, #f48fb1);
        color: #4a148c;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    /* Sidebar header */
    [data-testid="stSidebar"] h2 {
        color: #880e4f !important;
    }

    /* Buttons style */
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

    /* Code blocks style */
    .stCodeBlock pre {
        background-color: #fce4ec !important;
        color: #880e4f !important;
        font-family: 'Courier New', Courier, monospace !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }

    /* Table headers */
    .stTable thead tr th {
        background-color: #f48fb1 !important;
        color: #4a148c !important;
    }

    /* Info messages */
    .stInfo {
        background-color: #f8bbd0 !important;
        border-radius: 10px !important;
        color: #880e4f !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Database setup SQLite ---
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

def verify_user(username, password):
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    return c.fetchone() is not None

# --- Session state untuk login dan riwayat ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'history' not in st.session_state:
    st.session_state['history'] = []  # list simpan riwayat deteksi

# --- Fungsi login ---
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if verify_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success(f"Selamat datang, {username}!")
        else:
            st.error("Username atau password salah!")

# --- Sidebar untuk welcome dan logout ---
def sidebar_logged_in():
    st.sidebar.markdown(f"### Welcome, {st.session_state['username']} 👋")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = ''
        st.session_state['history'] = []

    st.sidebar.markdown("---")
    st.sidebar.caption("👩‍💻 Dikembangkan oleh Chisilia Amanda Wahyudi untuk skripsi deteksi penyakit buah jambu biji 🍈")

# --- Halaman utama setelah login ---
def home_page():
    st.title(f"Guava Disease Detection!")
    st.write("Selamat datang di aplikasi penerapan algoritma YOLOv11 untuk deteksi penyakit pada buah jambu biji berbasis web!")
    
    st.markdown("""
    **Tentang Jambu Biji dan Penyakit Utamanya**

    Jambu biji (Psidium guajava) adalah buah tropis kaya vitamin C dan senyawa bioaktif yang berperan penting dalam kesehatan. Selain sebagai sumber nutrisi, jambu biji memiliki manfaat obat tradisional seperti antibakteri dan antiradang.

    Namun, tanaman ini rentan terhadap tiga penyakit utama yang dapat mengurangi hasil panen:

    1. **Phytophthora**: Jamur yang menyebabkan busuk akar dan batang, menghambat penyerapan nutrisi, dan dapat menyebabkan kematian tanaman.
    2. **Scab**: Infeksi jamur yang menimbulkan bercak kasar coklat pada kulit buah dan daun, menurunkan kualitas dan nilai jual buah.
    3. **Styler and Root**: Menyerang putik bunga dan akar, menghambat pembentukan buah dan penyerapan nutrisi, sehingga menurunkan produksi.

    Penanganan efektif meliputi penggunaan varietas tahan penyakit, teknik budidaya yang baik, dan aplikasi fungisida yang tepat.
    """)

# --- Halaman deteksi jambu ---
def detection_page():
    st.title("🍎 Deteksi Penyakit Jambu menggunakan YOLOv11")

    ROOT = Path(__file__).parent
    IMAGES_DIR = ROOT / 'images'
    MODEL_DIR = ROOT / 'weights'

    DEFAULT_IMAGE = IMAGES_DIR / 'jambu1.jpg'
    DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detectedimage1.png'
    DETECTION_MODEL = MODEL_DIR / 'best.pt'

    st.sidebar.header("🔧 Konfigurasi Model")
    model_path = DETECTION_MODEL
    confidence_value = float(st.sidebar.slider("Confidence", 10, 100, 25)) / 100

    try:
        model = YOLO(model_path)
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
                    st.warning("Gambar default tidak ditemukan.")

        with col2:
            if st.sidebar.button("🔍 Deteksi Objek"):
                if not source_image and not DEFAULT_IMAGE.exists():
                    st.error("Tidak ada gambar untuk dideteksi.")
                    return

                result = model.predict(image, conf=confidence_value)
                boxes = result[0].boxes
                result_img = result[0].plot()[:, :, ::-1]  # RGB

                st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
                st.success(f"✅ Jumlah objek terdeteksi: {len(boxes)}")

                # Tampilkan hasil tensor bounding box seperti contoh
                st.write("### Results 1")
                boxes_str = str(boxes.xyxy.cpu())
                st.code(boxes_str, language="python")

                # Detail deteksi
                detections = []
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = model.names.get(cls_id, "Unknown")
                    conf = box.conf[0].item()
                    detections.append({"Label": label, "Confidence": f"{conf:.2f}"})

                if detections:
                    st.write("### 📋 Rincian Deteksi:")
                    st.table(detections)
                else:
                    st.write("Tidak ada objek yang terdeteksi.")

                # Simpan ke riwayat (dalam session_state)
                image_np = np.array(image.convert("RGB"))
                st.session_state['history'].append({
                    'type': 'Image',
                    'filename': source_image.name if source_image else 'default',
                    'detected_objects': len(boxes),
                    'input_img': image_np.tolist(),
                    'result_img': result_img.tolist(),
                    'detections': detections,
                    'boxes_tensor': boxes_str,  # simpan string tensor juga di riwayat
                })
            else:
                if DEFAULT_DETECT_IMAGE.exists():
                    st.image(DEFAULT_DETECT_IMAGE, caption="Deteksi Default", use_container_width=True)

    elif source_radio == "Video":
        video_file = st.sidebar.file_uploader("Unggah video...", type=["mp4", "mov", "avi"])
        if video_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame[..., ::-1], conf=confidence_value)
                result_frame = results[0].plot()
                stframe.image(result_frame, channels="RGB")

            cap.release()
        else:
            st.info("Unggah video untuk mulai deteksi.")

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
                    st.error("Gagal membaca frame dari kamera.")
                    break

                results = model(frame[..., ::-1], conf=confidence_value)
                result_frame = results[0].plot()
                FRAME_WINDOW.image(result_frame, channels="RGB")

            cap.release()

# --- Halaman Riwayat ---
def history_page():
    st.title("📜 Riwayat Deteksi")
    if not st.session_state['history']:
        st.info("Belum ada riwayat deteksi.")
    else:
        for i, item in enumerate(reversed(st.session_state['history']), 1):
            st.write(f"**{i}. Tipe:** {item['type']}, **File:** {item['filename']}, **Objek Terdeteksi:** {item['detected_objects']}")

            input_img_np = np.array(item['input_img'], dtype=np.uint8)
            st.image(input_img_np, caption=f"Gambar Input {item['filename']}", use_container_width=True)

            result_img_np = np.array(item['result_img'], dtype=np.uint8)
            st.image(result_img_np, caption=f"Hasil Deteksi {item['filename']}", use_container_width=True)

            st.write("#### Tensor Bounding Box:")
            st.code(item['boxes_tensor'], language="python")

            if item.get('detections'):
                st.write("#### 📋 Rincian Deteksi:")
                st.table(item['detections'])

# --- Main app logic ---
def main():
    girly_style()
    if not st.session_state['logged_in']:
        login()
    else:
        sidebar_logged_in()  # Tampilkan welcome + logout di sidebar
        menu = st.sidebar.radio("Menu", ["Home", "Detection", "Riwayat"])

        if menu == "Home":
            home_page()
        elif menu == "Detection":
            detection_page()
        elif menu == "Riwayat":
            history_page()

if __name__ == "__main__":
    main()

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

# Layout
st.set_page_config(
    page_title="Deteksi Penyakit Jambu Biji menggunakan YOLOv11",
    page_icon="🍐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auth (disimpan tapi dinonaktifkan)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT name, password FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user[0] if user and user[1] == hash_password(password) else None

# Inisialisasi sesi
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = True
    st.session_state['username'] = "admin"
    st.session_state['name'] = "Admin"

# Login
ENABLE_LOGIN = False

if ENABLE_LOGIN and st.session_state['authentication_status'] != True:
    st.header("Login 🍐")
    st.info("🔐 **Login Sementara**: Username = **admin**, Password = **123**")

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
    def show_mobile_warning():
        st.markdown("""
            <style>
            #mobile-warning {
                color: red;
                font-weight: bold;
                margin-top: 8px;
            }
            </style>
            <div id="mobile-warning" style="display:none;">
                ⚠️ Pengguna HP wajib menggunakan mode desktop agar aplikasi berfungsi dengan baik.
            </div>
            <script>
            function isMobile() {
                return /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            }
            if(isMobile()){
                document.getElementById("mobile-warning").style.display = "block";
            }
            </script>
        """, unsafe_allow_html=True)

    def main():
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False

        st.sidebar.title(f"Welcome")
        st.sidebar.header("🍐 Jambu Biji Indonesia")

        menu_options = {
            "Home": "🏠 Beranda",
            "Detection": "🔍 Deteksi Penyakit",
            "History": "📜 Riwayat"
        }
        selected_menu = st.sidebar.radio("Pilih Menu", list(menu_options.keys()), format_func=lambda x: menu_options[x])

        if selected_menu == "Home":
            st.header("Selamat datang di aplikasi YOLOv11 untuk deteksi penyakit jambu biji!")
            st.markdown(""" 
            Aplikasi ini mendeteksi penyakit pada buah jambu biji menggunakan model YOLOv11. Mode utama yang tersedia:  
            **Deteksi Gambar** – Mendeteksi dan memberi bounding box pada penyakit di buah jambu.  
            Silakan pilih mode yang Anda inginkan melalui menu di sidebar.
            """)

            col1, col2 = st.columns(2)
            with col1:
                st.image("images/jambu1.jpg", caption="Contoh Buah Jambu", use_column_width=True)
            with col2:
                st.image("images/detectedimage1.png", caption="Contoh Hasil Deteksi", use_column_width=True)

        elif selected_menu == "Detection":
            confidence = 0.4
            model_path = Path(settings.DETECTION_MODEL)

            try:
                model = helper.load_model(model_path)
            except Exception as ex:
                st.error(f"Tidak bisa memuat model: {model_path}")
                st.error(ex)
                return

            st.sidebar.header("Metode Input")
            input_method = st.sidebar.radio("Pilih metode input gambar:", ["Upload Gambar", "Kamera Langsung"])

            if input_method == "Upload Gambar":
                source_img = st.file_uploader("Unggah gambar jambu biji..", type=("jpg", "jpeg", "png"))
                show_mobile_warning()

                if source_img:
                    img = PIL.Image.open(source_img)
                    if st.button("Deteksi Penyakit"):
                        res = model.predict(img, conf=confidence)
                        boxes = res[0].boxes
                        plotted = res[0].plot()[:, :, ::-1]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img, caption="Gambar Asli", use_column_width=True)
                        with col2:
                            st.image(plotted, caption="Hasil Deteksi", use_column_width=True)

                        try:
                            with open("penyakit_jambu_info.json", "r", encoding="utf-8") as f:
                                penyakit_info = json.load(f)
                            detected_labels = set()
                            for box in boxes:
                                cls = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                                label = model.names[cls]
                                detected_labels.add(label)

                            penjelasan_list = []
                            for label in detected_labels:
                                if label in penyakit_info:
                                    penjelasan_list.append(f"**{label}**: {penyakit_info[label]}")
                                else:
                                    penjelasan_list.append(f"**{label}**: Info tidak tersedia")

                            if 'history' not in st.session_state:
                                st.session_state.history = []
                            st.session_state.history.append({
                                "image": img,
                                "result": plotted,
                                "boxes": boxes,
                                "penjelasan": penjelasan_list
                            })

                            st.markdown("### 🧠 Penjelasan Penyakit Terdeteksi")
                            for p in penjelasan_list:
                                st.info(p)

                        except:
                            st.warning("File penyakit_jambu_info.json tidak ditemukan")

            elif input_method == "Kamera Langsung":
                camera_image = st.camera_input("Ambil Foto dengan Kamera")
                if camera_image:
                    camera_img = PIL.Image.open(camera_image)
                    res_cam = model.predict(camera_img, conf=confidence)
                    boxes_cam = res_cam[0].boxes
                    plotted_cam = res_cam[0].plot()[:, :, ::-1]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(camera_img, caption="Gambar dari Kamera", use_column_width=True)
                    with col2:
                        st.image(plotted_cam, caption="Hasil Deteksi dari Kamera", use_column_width=True)

                    try:
                        with open("penyakit_jambu_info.json", "r", encoding="utf-8") as f:
                            penyakit_info = json.load(f)
                        detected_labels = set()
                        for box in boxes_cam:
                            cls = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])
                            label = model.names[cls]
                            detected_labels.add(label)

                        penjelasan_list = []
                        for label in detected_labels:
                            if label in penyakit_info:
                                penjelasan_list.append(f"**{label}**: {penyakit_info[label]}")
                            else:
                                penjelasan_list.append(f"**{label}**: Info tidak tersedia")

                        if 'history' not in st.session_state:
                            st.session_state.history = []
                        st.session_state.history.append({
                            "image": camera_img,
                            "result": plotted_cam,
                            "boxes": boxes_cam,
                            "penjelasan": penjelasan_list
                        })

                        st.markdown("### 🧠 Penjelasan Penyakit Terdeteksi")
                        for p in penjelasan_list:
                            st.info(p)

                    except:
                        st.warning("File penyakit_jambu_info.json tidak ditemukan")

        elif selected_menu == "History":
            st.header("Riwayat Deteksi")
            if st.session_state.get('history'):
                for idx, rec in enumerate(st.session_state.history):
                    st.subheader(f"Riwayat {idx + 1}")
                    st.image(rec['image'], caption=f"Gambar {idx + 1}", use_column_width=True)
                    st.image(rec['result'], caption=f"Hasil {idx + 1}", use_column_width=True)
                    if 'boxes' in rec:
                        with st.expander(f"Detail Kotak Deteksi {idx + 1}"):
                            for box in rec['boxes']:
                                st.write(box.data)
                    if 'penjelasan' in rec:
                        with st.expander(f"Penjelasan Penyakit {idx + 1}"):
                            for p in rec['penjelasan']:
                                st.markdown(p)
            else:
                st.write("Belum ada riwayat deteksi.")

        st.sidebar.markdown("---")
        st.session_state.dark_mode = st.sidebar.checkbox('Dark Mode', value=st.session_state.dark_mode)
        if st.session_state.dark_mode:
            st.sidebar.markdown("""<p style="color:white; font-size:12px;">❗ Gunakan saat Streamlit dalam mode gelap ❗</p>""", unsafe_allow_html=True)
            st.markdown(""" 
            <style>
            [data-testid="stAppViewContainer"] {background-color:#1E1E1E; color:#FFF;}
            [data-testid="stSidebar"] {background-color:#333; color:#FFF;}
            [data-testid="stExpander"] {background-color:#2E2E2E;}
            </style>""", unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""<p style="color:black; font-size:12px;">❗ Gunakan saat Streamlit dalam mode terang ❗</p>""", unsafe_allow_html=True)
            st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] {background-color:#ffffe0; color:#000;}
            [data-testid="stSidebar"] {background-color:#FFA62F; color:#000;}
            </style>""", unsafe_allow_html=True)

        st.sidebar.image("images/poon.png", use_column_width=True)

    if __name__ == "__main__" or True:
        main()

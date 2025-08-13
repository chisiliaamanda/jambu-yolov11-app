import sqlite3
from datetime import datetime
import streamlit as st
import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import os
import numpy as np
import tempfile
import cv2
import io

# -----------------------
# Setup & util functions
# -----------------------
RIWAYAT_DIR = Path("riwayat")
RIWAYAT_DIR.mkdir(exist_ok=True)

DB_PATH = "db_jambu.sqlite"

def connect():
    return sqlite3.connect(DB_PATH)

def create_tables():
    conn = connect()
    c = conn.cursor()

    # Tabel users
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    # Tabel riwayat
    c.execute("""
    CREATE TABLE IF NOT EXISTS riwayat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_name TEXT,
        image_detected TEXT,
        hasil TEXT,
        rincian TEXT,
        waktu TEXT
    )
    """)

    conn.commit()
    conn.close()

# -----------------------
# Auth & DB operations
# -----------------------
def register_user(username, password):
    try:
        conn = connect()
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None

def simpan_riwayat_db(user_id, image_name, image_detected_path, hasil, rincian, waktu=None):
    conn = connect()
    c = conn.cursor()
    if waktu is None:
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO riwayat (user_id, image_name, image_detected, hasil, rincian, waktu)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, image_name, str(image_detected_path), hasil, rincian, waktu))
    conn.commit()
    conn.close()

def ambil_riwayat_db(user_id):
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT id, image_name, image_detected, hasil, rincian, waktu
        FROM riwayat
        WHERE user_id = ?
        ORDER BY id ASC
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# -----------------------
# UI style
# -----------------------
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

# -----------------------
# Sidebar & Auth UI
# -----------------------
def sidebar_header():
    st.sidebar.markdown("### Selamat datang 👋")
    st.sidebar.markdown("---")
    st.sidebar.caption("👩‍💻 Oleh Chisilia Amanda Wahyudi | Skripsi Deteksi Penyakit Jambu 🍈")

    if 'user_id' in st.session_state:  # Menampilkan tombol Logout jika sudah login
        if st.sidebar.button("Logout"):
            logout_user()

def logout_user():
    if 'user_id' in st.session_state:
        del st.session_state['user_id']
    if 'logged_in' in st.session_state:
        del st.session_state['logged_in']
    st.success("Anda telah logout.")
    st.rerun()

def login_page():
    st.title("🔑 Login")

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        user_id = login_user(username, password)
        if user_id:
            st.session_state['user_id'] = user_id
            st.session_state['logged_in'] = True
            st.success("Login berhasil!")
            st.rerun()
        else:
            st.error("Username atau Password salah.")

def register_page():
    st.title("📝 Register")

    username = st.text_input("Username", key="reg_username")
    password = st.text_input("Password", type="password", key="reg_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")

    if st.button("Register"):
        if password == confirm_password:
            if register_user(username, password):
                st.success("Registrasi berhasil!")
                st.session_state['user_id'] = login_user(username, password)
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Username sudah terdaftar.")
        else:
            st.error("Password tidak cocok.")

# -----------------------
# Home Page
# -----------------------
def home_page():
    st.title("🌳 Guava Disease Detector")

    if 'user_id' not in st.session_state:
        page = st.selectbox("Pilih halaman", ["Login", "Register"])
        if page == "Login":
            login_page()
        elif page == "Register":
            register_page()
        return

    st.markdown("""<style>.big-font { font-size:20px !important; }</style>""", unsafe_allow_html=True)
    st.markdown('<div class="big-font">Selamat datang di <b>Guava Disease Detector</b>! Aplikasi ini dirancang untuk membantu Anda mengidentifikasi penyakit pada buah <b>jambu biji</b> secara otomatis menggunakan teknologi <i>YOLOv11</i>.</div>', unsafe_allow_html=True)

    st.image("images/detection.png", caption="Contoh Deteksi Penyakit pada Jambu Biji", width=600)

    st.subheader("🧠 Tentang Aplikasi")
    st.write("""Aplikasi berbasis web yang memanfaatkan model YOLOv11 untuk mendeteksi penyakit pada permukaan buah jambu biji. Sistem mendukung input dari gambar maupun kamera.""")

    st.subheader("🔍 Fitur Utama")
    st.markdown(""" 
    - ✅ Deteksi cepat dan akurat menggunakan YOLOv11.
    - 🖼️ Tampilan hasil deteksi dengan bounding box dan confidence.
    - 🕒 Riwayat deteksi tersimpan per pengguna.
    """)

    st.subheader("📌 Cara Menggunakan")
    st.markdown("""
    1. Masuk ke menu **🔍 Deteksi**.
    2. Pilih metode input: upload atau kamera.
    3. Jalankan deteksi dan lihat hasilnya.
    4. Cek kembali melalui menu **📜 Riwayat**.
    """)

    st.subheader("📞 Tentang Pengembang")
    st.markdown("""
    - 👩‍💻 **Nama**: Chisilia Amanda  
    - 🏫 **Universitas**: Universitas Gunadarma  
    - 📧 **Email**: chisiliaamanda123@gmail.com  
    - 🗂️ **GitHub**: [chisiliaamanda/guava-disease-detector](https://github.com/chisiliaamanda/jambu-yolov11-app.git)
    """)

# -----------------------
# Detection helpers
# -----------------------
def extract_detection_summary(result):
    """
    Build a readable summary (hasil) and rincian from ultralytics result object.
    """
    try:
        r = result[0]
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", {})
        if boxes is None or len(boxes) == 0:
            return "Tidak terdeteksi", "[]"

        # boxes.data -> array of [x1,y1,x2,y2,conf,class]
        data = boxes.data.cpu().numpy() if hasattr(boxes.data, "cpu") else np.array(boxes.data)
        rincian_list = []
        for row in data:
            conf = float(row[4])
            cls_idx = int(row[5])
            cls_name = names.get(cls_idx, str(cls_idx))
            rincian_list.append(f"{cls_name}:{conf:.2f}")
        hasil = ", ".join([item.split(":")[0] for item in rincian_list])
        rincian = "; ".join(rincian_list)
        return hasil, rincian
    except Exception as e:
        return "Error", str(e)

def save_output_image(output_image_np, filename):
    """
    Save numpy image (BGR or RGB) to riwayat folder and return path.
    """
    try:
        # if BGR, convert to RGB: try convert always but catch if not applicable
        if output_image_np is None:
            raise ValueError("No image to save")
        if len(output_image_np.shape) == 3 and output_image_np.shape[2] == 3:
            try:
                output_rgb = cv2.cvtColor(output_image_np, cv2.COLOR_BGR2RGB)
            except Exception:
                output_rgb = output_image_np
        else:
            output_rgb = output_image_np

        pil_img = Image.fromarray(output_rgb.astype("uint8"))
        save_path = RIWAYAT_DIR / filename
        pil_img.save(save_path)
        return save_path
    except Exception as e:
        st.error(f"Gagal menyimpan gambar: {e}")
        return None

# -----------------------
# Detection Page
# -----------------------
def detection_page():
    if 'user_id' not in st.session_state:
        st.error("Anda perlu login terlebih dahulu.")
        st.stop()

    st.title("🔍 Deteksi Penyakit pada Jambu Biji")

    metode = st.radio("Pilih Metode Input Gambar:", ["📁 Unggah Gambar", "📷 Gunakan Kamera"])
    confidence = st.slider("Tingkat Kepercayaan (Confidence)", 10, 100, 30) / 100

    # Lazy load model: hanya load ketika tombol deteksi ditekan
    model_path = "weights/best.pt"
    model = None

    image = None
    input_filename = None

    if metode == "📁 Unggah Gambar":
        uploaded_file = st.file_uploader("Unggah gambar jambu biji", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            input_filename = uploaded_file.name
            st.image(image, caption="Gambar Input", width=600)

    elif metode == "📷 Gunakan Kamera":
        camera_image = st.camera_input("Ambil gambar dengan kamera")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            input_filename = f"kamera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            st.image(image, caption="Gambar Input", width=600)

    if image is not None:
        if st.button("🔍 Jalankan Deteksi pada Gambar"):
            with st.spinner("Sedang mendeteksi..."):
                try:
                    # Convert PIL to numpy (BGR) for model if needed
                    img_np = np.array(image)
                    # ultralytics YOLO can accept numpy or PIL
                    if model is None:
                        model = YOLO(model_path)

                    result = model.predict(img_np, conf=confidence)

                    # get plotted image (array)
                    res_plotted = result[0].plot()

                    # show result
                    st.image(res_plotted, caption="Hasil Deteksi", use_column_width=True)

                    # extract summary
                    hasil, rincian = extract_detection_summary(result)

                    # save output image to riwayat
                    safe_name = f"deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{input_filename}"
                    save_path = save_output_image(res_plotted, safe_name)

                    # simpan ke DB
                    simpan_riwayat_db(
                        user_id=st.session_state['user_id'],
                        image_name=input_filename,
                        image_detected_path=save_path,
                        hasil=hasil,
                        rincian=rincian
                    )

                    st.success("Hasil deteksi berhasil disimpan ke riwayat.")
                except Exception as e:
                    st.error(f"Gagal mendeteksi: {e}")

# -----------------------
# History Page
# -----------------------
def history_page():
    if 'user_id' not in st.session_state:
        st.error("Anda perlu login terlebih dahulu.")
        st.stop()

    st.title("📜 Riwayat Deteksi")

    rows = ambil_riwayat_db(st.session_state['user_id'])
    if not rows:
        st.info("Belum ada deteksi yang disimpan.")
        return

    for i, row in enumerate(rows, 1):
        _id, image_name, image_detected, hasil, rincian, waktu = row
        st.subheader(f"Riwayat #{i} — {waktu}")
        if image_detected and os.path.exists(image_detected):
            st.image(image_detected, caption=f"{image_name} — {waktu}", use_column_width=True)
        else:
            st.warning("Gambar hasil deteksi tidak ditemukan pada path: " + str(image_detected))
        st.write(f"**Hasil Deteksi:** {hasil}")
        st.write(f"**Rincian:** {rincian}")
        st.markdown("---")

# -----------------------
# Main
# -----------------------
def main():
    create_tables()  # pastikan tabel tersedia
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

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

# Fungsi koneksi ke database
def connect():
    return sqlite3.connect("db_jambu.sqlite")

# Membuat tabel jika belum ada
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

    # Tabel riwayat (versi lengkap dengan image hasil deteksi dan rincian)
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

# Fungsi untuk registrasi user baru
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

# Fungsi login user
def login_user(username, password):
    conn = connect()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None

# Fungsi untuk menyimpan riwayat deteksi
def simpan_riwayat(user_id, image_name, image_detected, hasil, rincian, waktu=None):
    conn = connect()
    c = conn.cursor()
    if waktu is None:
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO riwayat (user_id, image_name, image_detected, hasil, rincian, waktu)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, image_name, image_detected, hasil, rincian, waktu))
    conn.commit()
    conn.close()

# Fungsi mengambil seluruh riwayat berdasarkan user_id
def ambil_riwayat(user_id):
    conn = connect()
    c = conn.cursor()
    c.execute("""
        SELECT id, image_detected, hasil, rincian, waktu 
        FROM riwayat 
        WHERE user_id = ? 
        ORDER BY id ASC
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return rows

# Function for styling the app with a girly theme
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

# Sidebar header content
def sidebar_header():
    st.sidebar.markdown("### Selamat datang 👋")
    st.sidebar.markdown("---")
    st.sidebar.caption("👩‍💻 Oleh Chisilia Amanda Wahyudi | Skripsi Deteksi Penyakit Jambu 🍈")

# Halaman Login
def login_page():
    st.title("🔑 Login")

    # Form login
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user_id = login_user(username, password)
        if user_id:
            st.session_state['user_id'] = user_id
            st.success("Login berhasil!")
            st.experimental_rerun()  # Refresh untuk akses ke halaman Deteksi
        else:
            st.error("Username atau Password salah.")

# Halaman Register
def register_page():
    st.title("📝 Register")

    # Form register
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password == confirm_password:
            if register_user(username, password):
                st.success("Registrasi berhasil!")
                st.session_state['user_id'] = login_user(username, password)  # Auto login setelah registrasi
                st.experimental_rerun()  # Refresh untuk akses ke halaman Deteksi
            else:
                st.error("Username sudah terdaftar.")
        else:
            st.error("Password tidak cocok.")

# Home page content
def home_page():
    st.title("🌳 Guava Disease Detector")
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
    - 🗂️ **GitHub**: [chisiliaamanda/guava-disease-detector](https://github.com/chisiliaamanda/guava-disease-detector)
    """)

# Detection page content
def detection_page():
    # Cek login terlebih dahulu
    if 'user_id' not in st.session_state:
        st.error("Anda perlu login terlebih dahulu.")
        st.stop()  # Menghentikan eksekusi jika belum login

    st.title("🔍 Deteksi Penyakit pada Jambu Biji")
    metode = st.radio("Pilih Metode Input Gambar:", ["📁 Unggah Gambar", "📷 Gunakan Kamera"])
    confidence = st.slider("Tingkat Kepercayaan (Confidence)", 10, 100, 30) / 100

    # Load the model
    model = YOLO("weights/best.pt")  # Gantilah dengan path yang sesuai untuk model YOLO

    image = None
    filename = None

    if metode == "📁 Unggah Gambar":
        uploaded_file = st.file_uploader("Unggah gambar jambu biji", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            filename = uploaded_file.name
    else:
        camera_image = st.camera_input("Ambil gambar dengan kamera")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            filename = f"kamera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    if image:
        st.image(image, caption="Gambar Input", width=600)

        if st.button("🔍 Jalankan Deteksi"):
            with st.spinner("Sedang mendeteksi..."):
                result = model.predict(image, conf=confidence)
                output_image = result[0].plot()[:, :, ::-1]
                st.image(output_image, caption="Hasil Deteksi", width=600)

                boxes = result[0].boxes
                labels = []
                label_details = []

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = model.model.names.get(cls_id, "Unknown")
                    conf_score = box.conf[0].item() * 100
                    labels.append(label)
                    label_details.append(f"- {label} ({conf_score:.2f}%)")

                hasil_deteksi = ", ".join(set(labels)) if labels else "Tidak ada penyakit terdeteksi"
                total_objek = len(labels)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.success(f"Hasil Deteksi: {hasil_deteksi}")
                st.info(f"Jumlah Objek Terdeteksi: {total_objek}")
                if labels:
                    st.write("📋 Rincian Deteksi:")
                    for item in label_details:
                        st.markdown(item)
                st.write(f"🕒 Waktu Deteksi: {timestamp}")

                # Simpan hasil ke folder riwayat
                RIWAYAT_FOLDER = Path(__file__).parent / "riwayat"
                if not RIWAYAT_FOLDER.exists():
                    RIWAYAT_FOLDER.mkdir(parents=True)

                image_path = os.path.join(RIWAYAT_FOLDER, filename)
                image.save(image_path)

                # Simpan riwayat ke database
                if 'user_id' in st.session_state:
                    simpan_riwayat(st.session_state['user_id'], filename, hasil_deteksi, timestamp)
                    st.info("Riwayat deteksi telah disimpan.")

# History page content
def history_page():
    # Cek login terlebih dahulu
    if 'user_id' not in st.session_state:
        st.error("Anda perlu login terlebih dahulu.")
        st.stop()  # Menghentikan eksekusi jika belum login

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

# Main function to run the app
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

# Run the app
if __name__ == "__main__":
    main()

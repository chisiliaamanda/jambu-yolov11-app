import sqlite3
from datetime import datetime
from pathlib import Path
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from ultralytics import YOLO

# --------- Konfigurasi dasar ----------
st.set_page_config(
    page_title="Guava Disease Detector",
    page_icon="🍈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ukuran gambar
INPUT_IMG_WIDTH = 380
RESULT_IMG_WIDTH = 520
HISTORY_IMG_WIDTH = 320
HOME_DEMO_WIDTH = 520

RIWAYAT_DIR = Path("riwayat")
RIWAYAT_DIR.mkdir(exist_ok=True)

DB_PATH = "db_jambu.sqlite"
WEIGHTS_PATH = Path("weights/best.pt")

# --------- Util umum ----------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --------- Normalisasi label & info penyakit ----------
def normalize_label(raw: str) -> str:
    key = (raw or "").strip().lower()
    aliases = {
        "phytophthora": "Phytophthora",
        "phytopthora": "Phytophthora",
        "phytohpthora": "Phytophthora",
        "phyptophthora": "Phytophthora",

        "styler": "Styler & Root",
        "root": "Styler & Root",
        "styler and root": "Styler & Root",
        "styler_root": "Styler & Root",
        "styler-root": "Styler & Root",
        "styler&root": "Styler & Root",

        "scab": "Scab",
    }
    return aliases.get(key, raw.strip() or "Tidak diketahui")

PENYAKIT_INFO = {
    "Phytophthora": (
        "Penyakit ini disebabkan oleh jamur *Phytophthora* yang menyebabkan bercak coklat "
        "kehitaman pada jambu biji dan dapat menimbulkan busuk basah."
    ),
    "Styler & Root": (
        "Kelompok gejala pada bagian stylar buah dan/atau akar. Stylar end rot ditandai kerusakan "
        "di ujung stylar buah, sedangkan root rot menyerang akar sehingga tanaman mudah layu."
    ),
    "Scab": "Scab menyebabkan bintik-bintik kasar pada kulit buah dan menurunkan kualitas visual.",
}

# --------- DB ----------
def connect():
    return sqlite3.connect(DB_PATH)

def create_tables():
    conn = connect()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS riwayat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        image_name TEXT,
        image_detected TEXT,
        hasil TEXT,
        rincian TEXT,
        waktu TEXT
    )""")
    conn.commit(); conn.close()

def register_user(username, password):
    try:
        conn = connect(); c = conn.cursor()
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit(); return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = connect(); c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username = ?", (username,))
    row = c.fetchone(); conn.close()
    if not row: return None
    user_id, pw_stored = row
    return user_id if pw_stored == password else None

def simpan_riwayat_db(user_id, image_name, image_detected_path, hasil, rincian, waktu=None):
    conn = connect(); c = conn.cursor()
    if waktu is None: waktu = now_str()
    c.execute("""
        INSERT INTO riwayat (user_id, image_name, image_detected, hasil, rincian, waktu)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, image_name, str(image_detected_path), hasil, rincian, waktu))
    conn.commit(); conn.close()

def ambil_riwayat_db(user_id):
    conn = connect(); c = conn.cursor()
    c.execute("""
        SELECT id, image_name, image_detected, hasil, rincian, waktu
        FROM riwayat WHERE user_id = ? ORDER BY id DESC
    """, (user_id,))
    rows = c.fetchall(); conn.close(); return rows

# --------- Styling UI ----------
def girly_style():
    st.markdown("""
    <style>
      /* 🌸 Background utama */
      .stApp {
        background: linear-gradient(135deg, #ffdde1, #ee9ca7) !important;
      }
      .block-container { 
        background: transparent !important; 
        padding: 2rem; 
      }

      /* Sidebar */
      [data-testid="stSidebar"] { 
        background: linear-gradient(135deg, #f48fb1, #ff80ab) !important; 
        color: white !important;
      }

      /* Input, widget, dll */
      .stTextInput > div > div > input,
      .stSelectbox > div > div,
      .stFileUploader,
      .stRadio > div,
      .stMarkdown,
      .stExpander,
      .stTextArea > div > textarea {
        background-color: rgba(255, 255, 255, 0.7) !important; 
        color: #4a0d36 !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.6rem !important;
      }

      /* Tombol */
      .stButton>button {
        background: linear-gradient(135deg, #ff80ab, #f48fb1) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: bold !important;
        padding: 0.5rem 1.2rem !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
      }
      .stButton>button:hover {
        background: linear-gradient(135deg, #f06292, #ec407a) !important;
        transform: scale(1.03);
        transition: 0.2s ease-in-out;
      }

      /* Font besar */
      .big-font { 
        font-size: 1.15rem; 
        line-height: 1.6; 
        color: #4a0d36;
      }

      /* Expander */
      .stExpander {
        background: rgba(255, 192, 224, 0.4) !important;
        border: none !important;
        border-radius: 12px !important;
      }
    </style>
    """, unsafe_allow_html=True)


# --------- Auth UI ----------
def logout_user():
    st.session_state.pop("user_id", None)
    st.session_state.pop("logged_in", None)
    st.success("Anda telah logout.")
    st.rerun()

def sidebar_header():
    st.sidebar.markdown("### Selamat datang 👋")
    st.sidebar.markdown("---")
    st.sidebar.caption("👩‍💻 Oleh Chisilia Amanda Wahyudi | Skripsi Deteksi Penyakit Jambu 🍈")
    if st.session_state.get("logged_in"):
        if st.sidebar.button("Logout"): logout_user()

def login_page():
    st.title("🔑 Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        user_id = login_user(username, password)
        if user_id:
            st.session_state["user_id"] = user_id
            st.session_state["logged_in"] = True
            st.success("Login berhasil!"); st.rerun()
        else:
            st.error("Username atau password salah.")

def register_page():
    st.title("📝 Register")
    username = st.text_input("Username", key="reg_username")
    password = st.text_input("Password", type="password", key="reg_password")
    confirm_password = st.text_input("Konfirmasi Password", type="password", key="reg_confirm")
    if st.button("Register"):
        if not username or not password:
            st.error("Username dan password wajib diisi."); return
        if password != confirm_password:
            st.error("Konfirmasi password tidak cocok."); return
        if register_user(username, password):
            st.success("Registrasi berhasil! Silakan login.")
        else:
            st.error("Username sudah terdaftar.")

# --------- Home ----------
def home_page():
    st.title("🌳 Guava Disease Detector")

    # kalau belum login -> tampilkan dropdown Login/Register
    if not st.session_state.get("logged_in"):
        pilihan = st.selectbox("Silakan pilih menu:", ["🔑 Login", "📝 Register"])
        if pilihan == "🔑 Login":
            login_page()
        else:
            register_page()
        return

    # kalau sudah login -> tampilkan halaman utama
    st.markdown(
        '<div class="big-font">Selamat datang di <b>Guava Disease Detector</b>! '
        'Aplikasi ini mengidentifikasi penyakit pada <b>jambu biji</b> menggunakan <i>YOLOv11</i>.</div>',
        unsafe_allow_html=True,
    )
    demo_img = Path("images/detection.png")
    if demo_img.exists():
        st.image(str(demo_img), caption="Contoh Deteksi Penyakit pada Jambu Biji", width=HOME_DEMO_WIDTH)

    st.subheader("🧠 Tentang Aplikasi")
    st.write("Mendeteksi penyakit pada permukaan buah jambu biji dari unggahan atau kamera.")
    st.subheader("🔍 Fitur Utama")
    st.markdown("- ✅ YOLOv11\n- 🖼️ Bounding box + confidence\n- 🕒 Riwayat per pengguna\n- 📷 Upload atau kamera")
    st.subheader("📌 Cara Menggunakan")
    st.markdown("1) Buka **Detection** · 2) Pilih sumber gambar · 3) Jalankan deteksi · 4) Cek **History**")
    with st.expander("ℹ️ Info Singkat Penyakit"):
        st.markdown("- **Phytophthora**: bercak coklat kehitaman\n- **Styler & Root**: di ujung stylar/akar\n- **Scab**: bintik kasar")

# --------- NMS sederhana ----------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a[:4]; bx1, by1, bx2, by2 = b[:4]
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2-inter_x1), max(0, inter_y2-inter_y1)
    inter = iw * ih
    area_a = (ax2-ax1)*(ay2-ay1); area_b = (bx2-bx1)*(by2-by1)
    union = area_a + area_b - inter + 1e-6
    return inter/union

def nms_merge(boxes, thr=0.6):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    for b in boxes:
        if all(iou_xyxy(b, k) < thr for k in keep):
            keep.append(b)
    return keep

# --------- Penyaring warna/tekstur ringan ----------
def _clip_int(v, lo, hi): 
    return int(max(lo, min(hi, v)))

def color_texture_filter(img_np, boxes):
    """Buang boks yang terlalu terang/halus/warna tak cocok (kurangi FP)."""
    if not boxes:
        return boxes
    try:
        import cv2
    except Exception:
        return boxes

    H, W = img_np.shape[:2]
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    filtered = []
    for (x1, y1, x2, y2, conf, name) in boxes:
        xi1 = max(0, min(W-1, int(x1))); yi1 = max(0, min(H-1, int(y1)))
        xi2 = max(1, min(W,   int(x2))); yi2 = max(1, min(H,   int(y2)))
        if xi2 <= xi1 or yi2 <= yi1:
            continue

        crop_hsv = img_hsv[yi1:yi2, xi1:xi2]
        crop_gray = cv2.cvtColor(img_bgr[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2GRAY)

        h = crop_hsv[...,0].astype(np.float32) / 180.0
        s = crop_hsv[...,1].astype(np.float32) / 255.0
        v = crop_hsv[...,2].astype(np.float32) / 255.0
        h_mean, s_mean, v_mean = float(h.mean()), float(s.mean()), float(v.mean())
        lap_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())

        ok = True
        if name == "Phytophthora":
            # dominan coklat (h ~ 0.05–0.20), cukup gelap & bertekstur
            if not (0.05 <= h_mean <= 0.20): ok = False
            if v_mean > 0.85: ok = False
            if s_mean < 0.12: ok = False
            if lap_var < 15:  ok = False
        elif name == "Scab":
            if s_mean < 0.18 or lap_var < 14: ok = False
        elif name == "Styler & Root":
            if lap_var < 12: ok = False

        if ok:
            filtered.append((x1, y1, x2, y2, conf, name))
    return filtered

# --------- Helper Deteksi ----------
def filter_predictions(result, img_w, img_h, base_conf=0.25):
    """
    Moderat-ketat: tangkap bercak jelas, buang yang ragu.
    """
    # CONF per kelas lebih tinggi
    per_class_conf = {
        "Phytophthora": max(base_conf, 0.34),
        "Scab":         max(base_conf, 0.36),
        "Styler & Root":max(base_conf, 0.40),
    }
    # Buang boks terlalu kecil + boks besar yang confidence rendah
    MIN_REL = 0.0030   # sebelumnya 0.0015
    BIG_REL = 0.35
    BIG_NEEDS = 0.65   # sebelumnya 0.55

    r = result[0]
    boxes = getattr(r, "boxes", None)
    names = getattr(r, "names", {}) or getattr(result, "names", {})
    if boxes is None or len(boxes) == 0:
        return []

    data = boxes.data.cpu().numpy() if hasattr(boxes.data, "cpu") else np.array(boxes.data)
    img_area = float(img_w * img_h)
    out = []

    for row in data:
        x1, y1, x2, y2, conf, cls_idx = row
        raw_name = names.get(int(cls_idx), str(int(cls_idx)))
        name = normalize_label(raw_name)

        thr = per_class_conf.get(name, base_conf)
        if conf < thr:
            continue

        area = max(0.0, (x2 - x1) * (y2 - y1))
        rel = area / img_area
        if rel < MIN_REL:
            continue
        if rel > BIG_REL and conf < BIG_NEEDS:
            continue

        out.append((float(x1), float(y1), float(x2), float(y2), float(conf), name))

    # NMS lebih agresif untuk hapus duplikat/overlap
    out = nms_merge(out, thr=0.50)
    return out

def draw_boxes_pil(image_rgb, boxes):
    if not isinstance(image_rgb, Image.Image):
        image_rgb = Image.fromarray(image_rgb)
    im = image_rgb.copy(); draw = ImageDraw.Draw(im)
    try: font = ImageFont.load_default()
    except Exception: font = None
    for (x1, y1, x2, y2, conf, name) in boxes:
        draw.rectangle([(x1, y1), (x2, y2)], outline="yellow", width=3)
        label = f"{name} {conf:.2f}"
        if hasattr(draw, "textbbox"): tw, th = draw.textbbox((0,0), label, font=font)[2:]
        else: tw, th = (len(label)*6, 12)
        pad = 2
        draw.rectangle([(x1, max(0, y1 - th - 2*pad)), (x1 + tw + 2*pad, y1)], fill="yellow")
        draw.text((x1 + pad, max(0, y1 - th - pad)), label, fill="black", font=font)
    return im

def save_output_image(output_image_np_or_pil, filename: str) -> Path | None:
    try:
        if output_image_np_or_pil is None: raise ValueError("No image to save")
        if isinstance(output_image_np_or_pil, Image.Image):
            img_pil = output_image_np_or_pil
        else:
            arr = output_image_np_or_pil
            if not isinstance(arr, np.ndarray): raise ValueError("Unsupported image type")
            if arr.dtype != np.uint8: arr = arr.astype("uint8")
            img_pil = Image.fromarray(arr)
        save_path = RIWAYAT_DIR / filename
        img_pil.save(save_path); return save_path
    except Exception as e:
        st.error(f"Gagal menyimpan gambar: {e}"); return None

# --------- Halaman Detection ----------
def detection_page():
    if not st.session_state.get("logged_in"):
        st.error("Anda perlu login terlebih dahulu."); st.stop()

    st.title("🔍 Deteksi Penyakit pada Jambu Biji")

    if not WEIGHTS_PATH.exists():
        st.error("Model tidak ditemukan. Pastikan file ada di: weights/best.pt"); st.stop()

    if "yolo_model" not in st.session_state:
        st.session_state["yolo_model"] = YOLO(str(WEIGHTS_PATH))
    model: YOLO = st.session_state["yolo_model"]

    metode = st.radio("Pilih Metode Input Gambar:", ["📁 Unggah Gambar", "📷 Gunakan Kamera"])
    # slider yang wajar; default 0.25
    base_conf = st.slider("Tingkat Kepercayaan (Confidence)", 10, 100, 32) / 100.0

    image = None; input_filename = None
    if metode == "📁 Unggah Gambar":
        uploaded_file = st.file_uploader("Unggah gambar jambu biji", type=["jpg","jpeg","png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            input_filename = uploaded_file.name
            st.image(image, caption="Gambar Input", width=INPUT_IMG_WIDTH)
    else:
        camera_image = st.camera_input("Ambil gambar dengan kamera")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            input_filename = f"kamera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            st.image(image, caption="Gambar Input", width=INPUT_IMG_WIDTH)

    if image is not None:
        if st.button("🔎 Jalankan Deteksi pada Gambar"):
            with st.spinner("Sedang mendeteksi..."):
                try:
                    img_np = np.array(image); H, W = img_np.shape[:2]

                    # Inference stabil (tidak agresif, tapi cukup detail)
                    result = model.predict(
                        img_np,
                        conf=base_conf,   # digabung dengan per-kelas
                        iou=0.45,         # lebih menekan overlap (lebih sedikit boks)
                        imgsz=1280,       # tetap detail tanpa over
                        augment=False,
                        agnostic_nms=False,
                        max_det=200,
                        verbose=False
                    )

                    boxes = filter_predictions(result, W, H, base_conf=base_conf)
                    boxes = color_texture_filter(img_np, boxes)   # saringan ringan anti-FP
                    boxes = nms_merge(boxes, thr=0.6)             # dedupe akhir

                    if not boxes:
                        st.warning("Tidak terdeteksi objek. Coba turunkan sedikit confidence atau ambil foto lebih terang/dekat.")
                        safe_name = f"deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{input_filename or 'input'}.jpg"
                        save_path = save_output_image(img_np, safe_name)
                        simpan_riwayat_db(
                            st.session_state["user_id"], input_filename or "kamera",
                            save_path or "", "Tidak terdeteksi", "[]"
                        )
                        st.info("Hasil (kosong) disimpan ke riwayat.")
                        return

                    plotted = draw_boxes_pil(image, boxes)
                    st.image(plotted, caption="Hasil Deteksi", width=RESULT_IMG_WIDTH)

                    # Ringkasan + rincian
                    kelas_list = [b[5] for b in boxes]
                    kelas_unique = []
                    for k in kelas_list:
                        if k not in kelas_unique: kelas_unique.append(k)
                    hasil = ", ".join(kelas_unique)

                    rincian_lines = []
                    for i, (x1, y1, x2, y2, conf, name) in enumerate(boxes, 1):
                        area = int((x2 - x1) * (y2 - y1))
                        rincian_lines.append(
                            f"Objek #{i}: {name} | Confidence: {conf:.2f} | "
                            f"Koordinat: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}] | "
                            f"Luas Box: {area} px²"
                        )
                    rincian = "\n".join(rincian_lines)

                    st.subheader("📋 Rincian Deteksi")
                    st.write(f"**Penyakit Terdeteksi:** {hasil}")
                    for line in rincian_lines: st.markdown(f"- {line}")
                    if hasil:
                        for cls_name in kelas_unique:
                            info = PENYAKIT_INFO.get(cls_name, "Informasi penyakit tidak tersedia.")
                            st.info(f"**{cls_name}:** {info}")

                    # Simpan hasil
                    safe_name = f"deteksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{input_filename or 'input'}.jpg"
                    save_path = save_output_image(plotted, safe_name)
                    simpan_riwayat_db(
                        st.session_state["user_id"], input_filename or "kamera",
                        save_path or "", hasil or "Tidak terdeteksi", rincian or "[]"
                    )
                    st.success("Hasil deteksi berhasil disimpan ke riwayat.")
                except Exception as e:
                    st.error(f"Gagal mendeteksi: {e}")

# --------- Halaman History ----------
def history_page():
    if not st.session_state.get("logged_in"):
        st.error("Anda perlu login terlebih dahulu."); st.stop()

    st.title("📜 Riwayat Deteksi")
    rows = ambil_riwayat_db(st.session_state["user_id"])
    if not rows:
        st.info("Belum ada deteksi yang disimpan."); return

    for i, row in enumerate(rows, 1):
        _id, image_name, image_detected, hasil, rincian, waktu = row
        st.subheader(f"Riwayat #{i} — {waktu}")
        if image_detected and os.path.exists(image_detected):
            st.image(image_detected, caption=f"{image_name} — {waktu}", width=HISTORY_IMG_WIDTH)
        else:
            st.warning("Gambar hasil deteksi tidak ditemukan pada path: " + str(image_detected))
        st.write(f"**Hasil Deteksi:** {hasil}")
        with st.expander("Lihat rincian"): st.text(rincian)
        st.markdown("---")

# --------- MAIN ----------
def main():
    create_tables(); girly_style(); sidebar_header()
    menu = st.sidebar.radio("📌 Menu", ["Home", "Detection", "History"], index=0)
    if menu == "Home": home_page()
    elif menu == "Detection": detection_page()
    elif menu == "History": history_page()

if __name__ == "__main__":
    main()

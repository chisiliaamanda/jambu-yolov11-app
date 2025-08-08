import sqlite3
from datetime import datetime

DB_NAME = "db_jambu.sqlite"

# Fungsi koneksi ke database
def connect():
    try:
        conn = sqlite3.connect(DB_NAME)
        print(f"Berhasil terhubung ke database: {DB_NAME}")
        return conn
    except sqlite3.Error as e:
        print(f"Gagal terhubung ke database: {e}")
        return None

# Membuat tabel jika belum ada
def create_tables():
    conn = connect()
    if conn:
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
        print("Tabel berhasil dibuat atau sudah ada.")  # Menambahkan feedback
    else:
        print("Koneksi database gagal, tabel tidak dapat dibuat.")

# Jalankan fungsi create_tables untuk inisialisasi database
create_tables()

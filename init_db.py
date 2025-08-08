import sqlite3
from datetime import datetime

DB_NAME = "db_jambu.sqlite"

# Fungsi koneksi ke database
def connect():
    return sqlite3.connect(DB_NAME)

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
    print("Tabel berhasil dibuat atau sudah ada.")  # Menambahkan feedback

# Jalankan fungsi create_tables untuk inisialisasi database
create_tables()

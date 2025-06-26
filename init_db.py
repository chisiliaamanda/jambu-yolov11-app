import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
''')

# Hapus user lama (opsional)
c.execute('DELETE FROM users')

# Tambah user default
users = [
    ('admin', '123'),
    ('user1', 'pass1'),
]

c.executemany('INSERT INTO users (username, password) VALUES (?, ?)', users)

conn.commit()
conn.close()
print("Database users.db sudah siap dengan user default.")

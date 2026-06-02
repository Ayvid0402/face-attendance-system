# database.py
# SQLite database for attendance storage
# More professional than CSV — supports queries and reporting

import sqlite3
import os
from datetime import datetime
import pandas as pd

# ── Setup ─────────────────────────────────────────────────────
DB_PATH = "attendance/attendance.db"

def create_database():
    """
    Creates the SQLite database and tables if they don't exist
    Think of this like creating an Excel file with column headers
    """
    # connect to database (creates file if doesn't exist)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # create attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            date        TEXT NOT NULL,
            time        TEXT NOT NULL,
            confidence  REAL DEFAULT 0.0,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # create persons table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT UNIQUE NOT NULL,
            registered  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("Database created successfully!")


def mark_attendance_db(name, confidence=0.0):
    """
    Mark attendance in SQLite database
    Prevents duplicate entries for same person on same day
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # check for duplicate
    cursor.execute('''
        SELECT id FROM attendance
        WHERE name = ? AND date = ?
    ''', (name, date))

    existing = cursor.fetchone()

    if existing:
        print(f"  {name} already marked present today!")
        conn.close()
        return False

    # insert new record
    cursor.execute('''
        INSERT INTO attendance (name, date, time, confidence)
        VALUES (?, ?, ?, ?)
    ''', (name, date, time, confidence))

    # add to persons table if not exists
    cursor.execute('''
        INSERT OR IGNORE INTO persons (name)
        VALUES (?)
    ''', (name,))

    conn.commit()
    conn.close()

    print(f"  DB: Attendance marked for {name} at {time}")
    return True


def get_attendance_db():
    """
    Get all attendance records as pandas dataframe
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT name, date, time, confidence
        FROM attendance
        ORDER BY date DESC, time DESC
    ''', conn)
    conn.close()
    return df


def get_today_attendance():
    """
    Get today's attendance only
    """
    conn = sqlite3.connect(DB_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    df = pd.read_sql_query('''
        SELECT name, date, time, confidence
        FROM attendance
        WHERE date = ?
        ORDER BY time
    ''', conn, params=(today,))
    conn.close()
    return df


def get_all_persons():
    """
    Get all registered persons
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT name, registered
        FROM persons
        ORDER BY name
    ''', conn)
    conn.close()
    return df


# ── Test the database ─────────────────────────────────────────
if __name__ == "__main__":
    print("Testing SQLite database...\n")

    # create database
    create_database()

    # test marking attendance
    mark_attendance_db("divya", confidence=83.1)
    mark_attendance_db("jeswin", confidence=79.5)
    mark_attendance_db("divya", confidence=83.1)  # duplicate!

    print("\nAll attendance records:")
    print(get_attendance_db())

    print("\nToday's attendance:")
    print(get_today_attendance())

    print("\nAll registered persons:")
    print(get_all_persons())

    print(f"\nDatabase saved to: {DB_PATH}")
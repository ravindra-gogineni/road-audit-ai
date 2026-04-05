import sqlite3
from datetime import datetime
import json

DB_NAME = "road_audits.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reporter_name TEXT,
            road_name TEXT,
            pothole_count INTEGER,
            total_cost INTEGER,
            status TEXT DEFAULT 'Pending',
            start_in_days INTEGER DEFAULT NULL,
            timestamp DATETIME,
            details_json TEXT,
            reporter_email TEXT,
            authority_email TEXT,
            priority TEXT DEFAULT 'Normal'
        )
    ''')
    # Backup: Add columns if they were missing (Schema Evolution)
    try:
        cursor.execute("ALTER TABLE complaints ADD COLUMN reporter_email TEXT")
        cursor.execute("ALTER TABLE complaints ADD COLUMN authority_email TEXT")
        cursor.execute("ALTER TABLE complaints ADD COLUMN priority TEXT")
    except:
        pass
    conn.commit()
    conn.close()

def add_complaint(reporter_name, road_name, pothole_count, total_cost, detections, reporter_email, authority_email, priority):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    details_json = json.dumps(detections)
    cursor.execute('''
        INSERT INTO complaints (reporter_name, road_name, pothole_count, total_cost, timestamp, details_json, reporter_email, authority_email, priority)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (reporter_name, road_name, pothole_count, total_cost, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), details_json, reporter_email, authority_email, priority))
    conn.commit()
    conn.close()

def get_all_complaints():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM complaints ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_complaint(complaint_id, status, start_in_days):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE complaints 
        SET status = ?, start_in_days = ?
        WHERE id = ?
    ''', (status, start_in_days, complaint_id))
    conn.commit()
    conn.close()

def delete_complaint(complaint_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM complaints WHERE id = ?', (complaint_id,))
    conn.commit()
    conn.close()

def search_complaints(query):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM complaints 
        WHERE road_name LIKE ? OR reporter_name LIKE ?
        ORDER BY timestamp DESC
    ''', (f'%{query}%', f'%{query}%'))
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print("Database initialized.")

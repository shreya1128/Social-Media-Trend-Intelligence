"""
database.py
-----------
SQLite database layer for the Social Media Trend Intelligence system.

Tables:
    users               -> basic authentication (demo account)
    prediction_history   -> every prediction made via the Prediction page

Keep intentionally minimal per project scope (no ORM, no migrations
framework - a small academic project does not need that complexity).
"""

import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(BASE_DIR, "database", "app.db")

DEMO_USERNAME = "admin"
DEMO_PASSWORD = "admin123"


class DatabaseError(Exception):
    """Raised for any SQLite failure, with a message safe to show a user
    (no raw stack trace / SQL is ever surfaced to the browser)."""
    pass


def get_connection():
    """Opens a connection. Raises DatabaseError with a friendly message on
    failure (e.g. the database/ folder is missing, disk is read-only, the
    file is corrupted, or it's locked by another process)."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn
    except (sqlite3.Error, OSError) as e:
        raise DatabaseError(
            "Could not connect to the database. Please check that the "
            "'database/' folder is writable and try again."
        ) from e


def init_db():
    """Creates tables if they don't exist, and seeds a demo login account."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                category TEXT NOT NULL,
                views INTEGER NOT NULL,
                likes INTEGER NOT NULL,
                comments INTEGER NOT NULL,
                shares INTEGER NOT NULL,
                growth_rate REAL NOT NULL,
                sentiment_score REAL NOT NULL,
                ds_prediction TEXT NOT NULL,
                ds_confidence REAL NOT NULL,
                dl_prediction TEXT NOT NULL,
                dl_confidence REAL NOT NULL,
                final_prediction TEXT,
                final_confidence REAL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

        # Lightweight migration: if an older app.db already exists without
        # final_prediction/final_confidence (from before the Prediction page
        # was changed to show one blended result instead of two separate
        # model outputs), add the columns without losing existing rows.
        cur.execute("PRAGMA table_info(prediction_history)")
        existing_cols = {row["name"] for row in cur.fetchall()}
        for col_name, col_type in [("final_prediction", "TEXT"), ("final_confidence", "REAL")]:
            if col_name not in existing_cols:
                cur.execute(f"ALTER TABLE prediction_history ADD COLUMN {col_name} {col_type}")
        conn.commit()

        # Seed demo user if not present
        cur.execute("SELECT id FROM users WHERE username = ?", (DEMO_USERNAME,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (DEMO_USERNAME, generate_password_hash(DEMO_PASSWORD)),
            )
            conn.commit()
            print(f"Seeded demo user -> username: '{DEMO_USERNAME}' password: '{DEMO_PASSWORD}'")
    except sqlite3.Error as e:
        raise DatabaseError("Failed to initialize the database schema.") from e
    finally:
        if conn is not None:
            conn.close()


def verify_user(username: str, password: str) -> bool:
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
    except sqlite3.Error as e:
        raise DatabaseError("Login is temporarily unavailable. Please try again.") from e
    finally:
        if conn is not None:
            conn.close()

    if row is None:
        return False
    return check_password_hash(row["password"], password)


def save_prediction(record: dict):
    """record must contain all prediction_history columns except id/created_at.
    Raises DatabaseError on failure; callers should treat a failed save as
    non-fatal (the prediction result itself can still be shown to the user)."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO prediction_history (
                platform, category, views, likes, comments, shares,
                growth_rate, sentiment_score,
                ds_prediction, ds_confidence, dl_prediction, dl_confidence,
                final_prediction, final_confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["platform"], record["category"], record["views"], record["likes"],
            record["comments"], record["shares"], record["growth_rate"], record["sentiment_score"],
            record["ds_prediction"], record["ds_confidence"], record["dl_prediction"],
            record["dl_confidence"], record["final_prediction"], record["final_confidence"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ))
        conn.commit()
    except (sqlite3.Error, KeyError) as e:
        raise DatabaseError("Could not save this prediction to the history log.") from e
    finally:
        if conn is not None:
            conn.close()


def get_recent_predictions(limit=10):
    """Returns [] (rather than raising) on failure, since the dashboard
    should still render even if the history log can't be read."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        # COALESCE handles rows saved before final_prediction/final_confidence
        # existed, so older history rows still display sensibly. Explicit
        # column list (rather than SELECT *) avoids a duplicate-name clash
        # with the aliased columns.
        cur.execute("""
            SELECT id, platform, category, views, likes, comments, shares,
                   growth_rate, sentiment_score,
                   ds_prediction, ds_confidence, dl_prediction, dl_confidence,
                   COALESCE(final_prediction, ds_prediction) AS final_prediction,
                   COALESCE(final_confidence, ds_confidence) AS final_confidence,
                   created_at
            FROM prediction_history ORDER BY id DESC LIMIT ?
        """, (limit,))
        return [dict(r) for r in cur.fetchall()]
    except (sqlite3.Error, DatabaseError):
        return []
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)

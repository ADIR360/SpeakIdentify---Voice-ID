import sqlite3
from typing import Dict, Optional, Tuple

import numpy as np


DB_PATH = 'voice_patterns.db'


def _ensure_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _get_table_columns(cursor: sqlite3.Cursor, table: str) -> set:
    cursor.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def create_database() -> None:
    """Create the database and the VoicePatterns table if they do not exist.

    Schema is standardized to use `name` (unique) and `pattern` (BLOB of float32 bytes).
    """
    conn = _ensure_connection()
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS VoicePatterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            pattern BLOB NOT NULL
        )
        '''
    )
    # New table to store multiple samples per user for centroiding/robustness
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS VoiceSamples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            pattern BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        '''
    )
    # Table to accumulate per-user acoustic statistics for voice conversion / visualization
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS UserStats (
            name TEXT PRIMARY KEY,
            mean_f0_sum REAL DEFAULT 0.0,
            mean_f0_count INTEGER DEFAULT 0,
            mfcc_mean_sum BLOB,
            mfcc_dim INTEGER DEFAULT 0,
            mfcc_count INTEGER DEFAULT 0
        )
        '''
    )
    conn.commit()

    # Migrate older schema: add `name` if table only has `letter`
    columns = _get_table_columns(cursor, 'VoicePatterns')
    if 'name' not in columns and 'letter' in columns:
        cursor.execute('ALTER TABLE VoicePatterns ADD COLUMN name TEXT')
        # Backfill name from letter
        cursor.execute('UPDATE VoicePatterns SET name = letter WHERE name IS NULL')
        # Ensure uniqueness on `name` if possible
        try:
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_voicepatterns_name ON VoicePatterns(name)')
        except sqlite3.OperationalError:
            pass
        conn.commit()
    elif 'name' in columns and 'letter' not in columns:
        # Fresh DB: create a unique index on name
        try:
            cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_voicepatterns_name ON VoicePatterns(name)')
        except sqlite3.OperationalError:
            pass

    conn.close()


def save_voice_pattern(name: str, voice_pattern: np.ndarray) -> None:
    """Insert or update a voice pattern for the given name.

    The pattern is stored as float32 bytes to keep storage small and consistent.
    """
    if not isinstance(voice_pattern, np.ndarray):
        voice_pattern = np.asarray(voice_pattern)
    voice_pattern = voice_pattern.astype(np.float32, copy=False)

    conn = _ensure_connection()
    cursor = conn.cursor()
    columns = _get_table_columns(cursor, 'VoicePatterns')

    if 'name' in columns and 'letter' in columns:
        # Legacy DB with both columns: upsert manually to satisfy NOT NULL on letter
        cursor.execute('SELECT rowid FROM VoicePatterns WHERE name = ? OR letter = ? LIMIT 1', (name, name))
        row = cursor.fetchone()
        if row:
            cursor.execute('UPDATE VoicePatterns SET name = ?, letter = ?, pattern = ? WHERE rowid = ?', (name, name, voice_pattern.tobytes(), row[0]))
        else:
            cursor.execute('INSERT INTO VoicePatterns (name, letter, pattern) VALUES (?, ?, ?)', (name, name, voice_pattern.tobytes()))
    elif 'name' in columns:
        cursor.execute(
            'INSERT INTO VoicePatterns (name, pattern) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET pattern=excluded.pattern',
            (name, voice_pattern.tobytes()),
        )
    elif 'letter' in columns:
        # Backward-only DB
        cursor.execute(
            'INSERT INTO VoicePatterns (letter, pattern) VALUES (?, ?) ON CONFLICT(letter) DO UPDATE SET pattern=excluded.pattern',
            (name, voice_pattern.tobytes()),
        )
    else:
        raise RuntimeError('VoicePatterns table has no compatible name/letter column')

    # Also append to samples table to build a history for robust averaging
    try:
        cursor.execute('INSERT INTO VoiceSamples (name, pattern) VALUES (?, ?)', (name, voice_pattern.tobytes()))
    except sqlite3.OperationalError:
        # Table may not exist on very old DBs; ignore silently
        pass

    conn.commit()
    conn.close()


def _decode_pattern(blob: bytes) -> np.ndarray:
    """Decode a stored pattern blob to float32 numpy array, tolerant to float64 legacy data."""
    nbytes = len(blob)
    # Try float32 first
    if nbytes % 4 == 0:
        arr32 = np.frombuffer(blob, dtype=np.float32)
        if arr32.size == 13:
            return arr32
    # Try float64 then cast
    if nbytes % 8 == 0:
        arr64 = np.frombuffer(blob, dtype=np.float64)
        return arr64.astype(np.float32, copy=False)
    # Fallback
    return np.frombuffer(blob, dtype=np.float32)


def _aggregate_centroids(cursor: sqlite3.Cursor) -> Dict[str, np.ndarray]:
    """Aggregate multiple samples per user from VoiceSamples into centroids."""
    centroids: Dict[str, np.ndarray] = {}
    try:
        cursor.execute('SELECT name, pattern FROM VoiceSamples')
        rows = cursor.fetchall()
        if not rows:
            return centroids
        # Accumulate sums and counts per name
        sums: Dict[str, np.ndarray] = {}
        counts: Dict[str, int] = {}
        for name, pattern in rows:
            vec = _decode_pattern(pattern)
            if name not in sums:
                sums[name] = vec.astype(np.float32, copy=False).copy()
                counts[name] = 1
            else:
                # Ensure shape alignment
                if sums[name].shape == vec.shape:
                    sums[name] += vec.astype(np.float32, copy=False)
                    counts[name] += 1
        for name, total in sums.items():
            centroids[name] = (total / max(1, counts[name])).astype(np.float32, copy=False)
    except sqlite3.OperationalError:
        # Table not present; skip
        pass
    return centroids


def load_database() -> Dict[str, np.ndarray]:
    """Load all voice patterns into a dictionary mapping name -> np.ndarray(float32).

    Preference order:
    1) Aggregated centroids from VoiceSamples if available
    2) Fallback to single template entries from VoicePatterns
    """
    conn = _ensure_connection()
    cursor = conn.cursor()

    # Try centroids first
    voice_db: Dict[str, np.ndarray] = _aggregate_centroids(cursor)
    if voice_db:
        conn.close()
        return voice_db

    try:
        cursor.execute('SELECT name, pattern FROM VoicePatterns')
        rows = cursor.fetchall()
        for name, pattern in rows:
            voice_db[name] = _decode_pattern(pattern)
    except sqlite3.OperationalError:
        # Backward compatibility for older DBs that used `letter`
        cursor.execute('SELECT letter, pattern FROM VoicePatterns')
        rows = cursor.fetchall()
        for letter, pattern in rows:
            voice_db[letter] = _decode_pattern(pattern)

    conn.close()
    return voice_db


def update_user_stats(name: str, mean_f0: Optional[float], mfcc_mean: Optional[np.ndarray]) -> None:
    """Update per-user acoustic statistics by adding a new observation.

    - mean_f0: average F0 (Hz) for a clip (None to skip)
    - mfcc_mean: 1D np.ndarray of mean MFCCs for a clip (None to skip)
    Accumulates running sums and counts to allow robust averaging.
    """
    conn = _ensure_connection()
    cursor = conn.cursor()
    # Ensure row exists
    cursor.execute('SELECT name FROM UserStats WHERE name = ?', (name,))
    exists = cursor.fetchone() is not None
    if not exists:
        cursor.execute('INSERT OR IGNORE INTO UserStats (name, mean_f0_sum, mean_f0_count, mfcc_mean_sum, mfcc_dim, mfcc_count) VALUES (?, 0.0, 0, ?, ?, 0)',
                       (name, (mfcc_mean.astype(np.float32).tobytes() if mfcc_mean is not None else None), (int(mfcc_mean.size) if mfcc_mean is not None else 0)))

    # Update mean_f0
    if mean_f0 is not None:
        cursor.execute('UPDATE UserStats SET mean_f0_sum = COALESCE(mean_f0_sum, 0.0) + ?, mean_f0_count = COALESCE(mean_f0_count, 0) + 1 WHERE name = ?',
                       (float(mean_f0), name))

    # Update MFCC sum and dim/count
    if mfcc_mean is not None:
        mfcc_mean = np.asarray(mfcc_mean, dtype=np.float32)
        # Read existing sum and dim
        cursor.execute('SELECT mfcc_mean_sum, mfcc_dim, mfcc_count FROM UserStats WHERE name = ?', (name,))
        row = cursor.fetchone()
        if row is not None:
            blob, dim, count = row
            if blob is None or dim is None or dim == 0:
                new_sum = mfcc_mean
                new_dim = int(mfcc_mean.size)
                new_count = 1
            else:
                try:
                    cur_sum = np.frombuffer(blob, dtype=np.float32)
                except Exception:
                    cur_sum = np.zeros_like(mfcc_mean, dtype=np.float32)
                    dim = mfcc_mean.size
                if int(dim) != int(mfcc_mean.size):
                    # Reset if dimension changed
                    new_sum = mfcc_mean
                    new_dim = int(mfcc_mean.size)
                    new_count = 1
                else:
                    new_sum = (cur_sum + mfcc_mean).astype(np.float32, copy=False)
                    new_dim = int(dim)
                    new_count = int(count or 0) + 1
            cursor.execute('UPDATE UserStats SET mfcc_mean_sum = ?, mfcc_dim = ?, mfcc_count = ? WHERE name = ?',
                           (new_sum.tobytes(), new_dim, new_count, name))
    conn.commit()
    conn.close()


def get_user_stats(name: str) -> Optional[Tuple[Optional[float], Optional[np.ndarray]]]:
    """Return (mean_f0_avg, mfcc_mean_avg) for a user if available, else None.

    - mean_f0_avg: float or None if no F0 data
    - mfcc_mean_avg: np.ndarray[float32] of shape (mfcc_dim,) or None
    """
    conn = _ensure_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT mean_f0_sum, mean_f0_count, mfcc_mean_sum, mfcc_dim, mfcc_count FROM UserStats WHERE name = ?', (name,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    f0_sum, f0_count, mfcc_blob, mfcc_dim, mfcc_count = row
    mean_f0_avg: Optional[float] = None
    if f0_count and f0_count > 0:
        mean_f0_avg = float(f0_sum) / float(f0_count)
    mfcc_mean_avg: Optional[np.ndarray] = None
    if mfcc_blob is not None and mfcc_dim and mfcc_count and mfcc_count > 0:
        arr = np.frombuffer(mfcc_blob, dtype=np.float32)
        # It's the running sum; average it
        mfcc_mean_avg = (arr / float(mfcc_count)).astype(np.float32, copy=False)
    return (mean_f0_avg, mfcc_mean_avg)


def get_all_user_stats() -> Dict[str, Tuple[Optional[float], Optional[np.ndarray]]]:
    """Return a mapping name -> (mean_f0_avg, mfcc_mean_avg) for all users with any stats."""
    conn = _ensure_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT name, mean_f0_sum, mean_f0_count, mfcc_mean_sum, mfcc_dim, mfcc_count FROM UserStats')
    rows = cursor.fetchall()
    conn.close()
    result: Dict[str, Tuple[Optional[float], Optional[np.ndarray]]] = {}
    for name, f0_sum, f0_count, mfcc_blob, mfcc_dim, mfcc_count in rows:
        mean_f0_avg: Optional[float] = None
        if f0_count and f0_count > 0:
            mean_f0_avg = float(f0_sum) / float(f0_count)
        mfcc_mean_avg: Optional[np.ndarray] = None
        if mfcc_blob is not None and mfcc_dim and mfcc_count and mfcc_count > 0:
            arr = np.frombuffer(mfcc_blob, dtype=np.float32)
            mfcc_mean_avg = (arr / float(mfcc_count)).astype(np.float32, copy=False)
        result[name] = (mean_f0_avg, mfcc_mean_avg)
    return result


def list_users() -> list:
    """Return a sorted list of unique user names present in VoiceSamples or VoicePatterns."""
    conn = _ensure_connection()
    cursor = conn.cursor()
    names = set()
    try:
        cursor.execute('SELECT DISTINCT name FROM VoiceSamples')
        names.update(n for (n,) in cursor.fetchall() if n)
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute('SELECT DISTINCT name FROM VoicePatterns')
        names.update(n for (n,) in cursor.fetchall() if n)
    except sqlite3.OperationalError:
        # legacy fallback
        try:
            cursor.execute('SELECT DISTINCT letter FROM VoicePatterns')
            names.update(n for (n,) in cursor.fetchall() if n)
        except sqlite3.OperationalError:
            pass
    conn.close()
    return sorted(names)


def delete_user(name: str) -> None:
    """Delete all samples and templates for a given user name."""
    conn = _ensure_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM VoiceSamples WHERE name = ?', (name,))
    except sqlite3.OperationalError:
        pass
    # Handle both modern and legacy columns
    columns = _get_table_columns(cursor, 'VoicePatterns')
    if 'name' in columns:
        cursor.execute('DELETE FROM VoicePatterns WHERE name = ?', (name,))
    if 'letter' in columns:
        cursor.execute('DELETE FROM VoicePatterns WHERE letter = ?', (name,))
    conn.commit()
    conn.close()


def rename_user(old_name: str, new_name: str) -> None:
    """Rename a user across all tables (atomic)."""
    if not new_name:
        return
    conn = _ensure_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('BEGIN')
        try:
            cursor.execute('UPDATE VoiceSamples SET name = ? WHERE name = ?', (new_name, old_name))
        except sqlite3.OperationalError:
            pass
        columns = _get_table_columns(cursor, 'VoicePatterns')
        if 'name' in columns:
            cursor.execute('UPDATE VoicePatterns SET name = ? WHERE name = ?', (new_name, old_name))
        if 'letter' in columns:
            cursor.execute('UPDATE VoicePatterns SET letter = ? WHERE letter = ?', (new_name, old_name))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

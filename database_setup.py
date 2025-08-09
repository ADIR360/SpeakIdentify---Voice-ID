import sqlite3


def create_database():
    conn = sqlite3.connect('voice_patterns.db')
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

    # Ensure a unique index on name for efficient upserts
    try:
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_voicepatterns_name ON VoicePatterns(name)')
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()


if __name__ == "__main__":
    create_database()

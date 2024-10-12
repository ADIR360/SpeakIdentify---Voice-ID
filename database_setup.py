import sqlite3

def create_database():
    conn = sqlite3.connect('voice_patterns.db')  # Connect to the database (or create it)
    cursor = conn.cursor()

    # Create a table to store voice patterns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS VoicePatterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            pattern BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()  # Run the function to create the database

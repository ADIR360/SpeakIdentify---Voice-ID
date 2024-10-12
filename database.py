import numpy as np
import sqlite3
import os
import librosa

def create_database():
    conn = sqlite3.connect('voice_patterns.db')  # Connect to the database (or create it)
    cursor = conn.cursor()

    # Create a table to store voice patterns if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS VoicePatterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            letter TEXT NOT NULL UNIQUE,
            pattern BLOB NOT NULL
        )
    ''')
    conn.commit()

    # Populate the database with voice patterns for each letter
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        audio_file_path = f"audio_samples/{letter}.wav"
        if os.path.exists(audio_file_path):
            audio_data, sample_rate = librosa.load(audio_file_path, sr=22050)
            voice_pattern = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13).T, axis=0)
            cursor.execute('''
                INSERT OR IGNORE INTO VoicePatterns (letter, pattern) VALUES (?, ?)
            ''', (letter, voice_pattern.tobytes()))  # Store the numpy array as bytes

    conn.commit()
    conn.close()

def load_database():
    conn = sqlite3.connect('voice_patterns.db')
    cursor = conn.cursor()

    # Fetch all voice patterns from the database
    cursor.execute('SELECT letter, pattern FROM VoicePatterns')
    voice_db = {}
    
    for row in cursor.fetchall():
        letter, pattern = row
        voice_db[letter] = np.frombuffer(pattern, dtype=np.float32)  # Convert BLOB back to numpy array

    conn.close()
    return voice_db

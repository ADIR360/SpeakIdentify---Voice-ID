import sounddevice as sd
import numpy as np
import librosa

def record_audio(duration=3, sample_rate=22050):
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float64')
    sd.wait()  # Wait for the recording to finish
    return audio_data.flatten()  # Flatten the array to 1D

def process_audio(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=13)  # Extract MFCC features
    return np.mean(mfccs.T, axis=0)  # Return mean MFCCs

def identify_speaker(processed_audio, voice_db):
    best_match = None
    best_score = float('inf')

    for name, voice_pattern in voice_db.items():
        score = np.linalg.norm(processed_audio - voice_pattern)  # Calculate similarity score
        if score < best_score:
            best_score = score
            best_match = name

    return best_match

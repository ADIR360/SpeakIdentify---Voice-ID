import sounddevice as sd
import numpy as np
import librosa


DEFAULT_SAMPLE_RATE = 16000


def record_audio(duration: float = 3.0, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Record mono audio and return a 1D float32 array at the given sample rate."""
    num_frames = int(duration * sample_rate)
    audio_data = sd.rec(
        num_frames,
        samplerate=sample_rate,
        channels=1,
        dtype='float32',
        blocking=False,
    )
    sd.wait()
    return audio_data.flatten()


def _cmvn(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Cepstral mean-variance normalization across time (axis=1 if (num_coeffs, num_frames))."""
    if features.ndim == 2:
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        std = np.where(std < eps, eps, std)
        return (features - mean) / std
    return features


def process_audio(audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Extract a compact MFCC embedding with CMVN and mean pooling as float32."""
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.asarray(audio_data)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32, copy=False)

    # Use smaller FFT and hop to reduce CPU while preserving intelligibility
    mfccs = librosa.feature.mfcc(
        y=audio_data,
        sr=sample_rate,
        n_mfcc=13,
        n_fft=512,
        hop_length=256,
        center=True,
    )
    mfccs = _cmvn(mfccs)
    embedding = np.mean(mfccs.T, axis=0).astype(np.float32, copy=False)
    return embedding

def _l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec) + eps
    return (vec / norm).astype(np.float32, copy=False)


def identify_speaker(processed_audio: np.ndarray, voice_db: dict, similarity_threshold: float = 0.6):
    """Identify speaker using cosine similarity with length-normalized embeddings.

    Returns the best-matching name if similarity >= threshold; otherwise None.
    """
    if not voice_db:
        return None

    q = _l2_normalize(processed_audio)
    best_name = None
    best_sim = -1.0

    for name, pattern in voice_db.items():
        p = _l2_normalize(pattern)
        sim = float(np.dot(q, p))  # cosine similarity on L2-normalized vectors
        if sim > best_sim:
            best_sim = sim
            best_name = name

    return best_name if best_sim >= similarity_threshold else None


def rank_similarities(processed_audio: np.ndarray, voice_db: dict, top_k: int = 5):
    """Return top-k matches as a list of (name, cosine_similarity) sorted desc."""
    if not voice_db:
        return []
    q = _l2_normalize(processed_audio)
    scores = []
    for name, pattern in voice_db.items():
        p = _l2_normalize(pattern)
        sim = float(np.dot(q, p))
        scores.append((name, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def compute_pitch_series(
    audio_data: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    fmin: float = 80.0,
    fmax: float = 400.0,
    frame_length: int = 1024,
    hop_length: int = 256,
):
    """Compute pitch (F0) time series using YIN.

    Returns (times, f0_hz) with unvoiced frames set to 0.
    """
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32, copy=False)

    f0 = librosa.yin(
        audio_data,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    ).astype(np.float32, copy=False)

    times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sample_rate,
        hop_length=hop_length,
    ).astype(np.float32, copy=False)

    # Replace NaNs with 0 for unvoiced
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32, copy=False)
    return times, f0


def estimate_ai_score(
    audio_data: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> float:
    """Heuristic AI-synthesis likelihood score in [0,1].

    Uses spectral flatness and pitch variability as simple cues. Not production-grade.
    """
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32, copy=False)

    # Spectral flatness (higher may indicate synthetic/over-smoothed spectra)
    S = np.abs(
        librosa.stft(
            audio_data,
            n_fft=frame_length,
            hop_length=hop_length,
            center=True,
        )
    )
    flatness = librosa.feature.spectral_flatness(S=S).mean()

    # Pitch variability (very low variance may indicate TTS)
    _, f0 = compute_pitch_series(audio_data, sample_rate, frame_length=frame_length, hop_length=hop_length)
    voiced = f0[f0 > 0]
    pitch_var = float(np.var(voiced)) if voiced.size > 10 else 0.0

    # Normalize heuristics to [0,1]
    flatness_score = float(np.clip((flatness - 0.2) / 0.6, 0.0, 1.0))
    pitch_uniformity_score = float(np.clip(1.0 - (pitch_var / (50.0 ** 2)), 0.0, 1.0))

    ai_score = 0.6 * flatness_score + 0.4 * pitch_uniformity_score
    return float(ai_score)

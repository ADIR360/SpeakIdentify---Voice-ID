import sounddevice as sd
import numpy as np
import librosa
from typing import Tuple, Optional
from scipy import signal


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


def compute_clip_stats(
    audio_data: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Tuple[Optional[float], np.ndarray]:
    """Compute simple per-clip statistics for voice conversion and visualization.

    Returns (mean_f0_hz or None, mfcc_mean_vector[13])
    """
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32, copy=False)

    # Mean F0 over voiced frames
    _, f0 = compute_pitch_series(audio_data, sample_rate)
    voiced = f0[f0 > 0]
    mean_f0 = float(np.mean(voiced)) if voiced.size >= 10 else None

    # Mean MFCC vector
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=512, hop_length=256, center=True)
    mfccs = _cmvn(mfccs)
    mfcc_mean = np.mean(mfccs, axis=1).astype(np.float32, copy=False)
    return mean_f0, mfcc_mean


def _apply_spectral_tilt(audio: np.ndarray, sample_rate: int, tilt_db_per_octave: float) -> np.ndarray:
    """Apply a simple spectral tilt using a first-order shelving filter.

    Positive tilt boosts lows; negative boosts highs.
    """
    # Design a low-shelf IIR using bilinear transform approximation
    # Use 1 kHz as shelf corner
    f0 = 1000.0
    # Convert tilt in dB/oct to gain at f0 relative to Nyquist (~one octave mapping heuristic)
    # Map ±12 dB/oct to ±9 dB at f0 as a rough scale
    gain_db = np.clip(tilt_db_per_octave * 0.75, -12.0, 12.0)
    G = 10 ** (gain_db / 20.0)
    w0 = 2 * np.pi * f0 / sample_rate
    # First-order shelf from RBJ cookbook
    alpha = np.sin(w0) / 2.0 * np.sqrt(2.0)
    A = np.sqrt(G)
    cosw0 = np.cos(w0)
    b0 =    A*((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 =  2*A*((A-1) - (A+1)*cosw0)
    b2 =    A*((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =   -2*((A-1) + (A+1)*cosw0)
    a2 =        (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha
    b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float32)
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float32)
    return signal.lfilter(b, a, audio).astype(np.float32, copy=False)


def voice_convert(
    audio_data: np.ndarray,
    source_stats: Tuple[Optional[float], Optional[np.ndarray]],
    target_stats: Tuple[Optional[float], Optional[np.ndarray]],
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    """Lightweight voice conversion.

    - Pitch shift to align mean F0 with target if both are available.
    - Apply gentle spectral tilt guided by MFCC[1] difference.
    Designed for efficiency, not perfect mimicry.
    """
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32, copy=False)

    src_f0, src_mfcc = source_stats
    tgt_f0, tgt_mfcc = target_stats
    y = audio_data

    # Pitch shift
    if src_f0 is not None and tgt_f0 is not None and src_f0 > 0 and tgt_f0 > 0:
        semitones = 12.0 * np.log2(float(tgt_f0) / float(src_f0))
        semitones = float(np.clip(semitones, -6.0, 6.0))
        try:
            y = librosa.effects.pitch_shift(y=y, sr=sample_rate, n_steps=semitones)
        except Exception:
            pass

    # Spectral tilt from MFCC[1] difference
    if src_mfcc is not None and tgt_mfcc is not None and src_mfcc.size >= 2 and tgt_mfcc.size >= 2:
        diff = float(tgt_mfcc[1] - src_mfcc[1])
        # Map MFCC diff to dB/oct tilt within a small safe range
        tilt = np.clip(diff * 2.0, -6.0, 6.0)
        try:
            y = _apply_spectral_tilt(y, sample_rate, tilt)
        except Exception:
            pass

    # Output normalization to prevent clipping
    peak = float(np.max(np.abs(y)) + 1e-8)
    if peak > 1.0:
        y = (y / peak).astype(np.float32, copy=False)
    return y


def compute_spectrogram(
    audio_data: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    """Return log-magnitude Mel spectrogram for visualization (dB)."""
    if not isinstance(audio_data, np.ndarray):
        audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32, copy=False)
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=64, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32, copy=False)
    return S_db

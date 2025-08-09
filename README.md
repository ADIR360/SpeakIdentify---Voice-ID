# SpeakIdentify — Voice ID

## Overview
SpeakIdentify is a local, privacy-friendly voice identification tool. It records short utterances, extracts compact embeddings, and matches them to enrolled users. It now includes threshold control, multi-sample enrollment, pitch and match visualizations, file import for public voices, and a basic AI-likelihood heuristic.

## File Structure
```
SpeakIdentify---Voice-ID/
  main.py               # Tkinter UI: controls, visualizations, user management
  audio_processing.py   # Recording, MFCC+CMVN features, cosine matching, pitch, AI heuristic
  database.py           # SQLite schema, migration, multi-sample storage, CRUD helpers
  database_setup.py     # Minimal DB bootstrap (aligned with `name` schema)
  requirements.txt      # Runtime dependencies
  README.md             # This document
  voice_patterns.db     # SQLite database (generated)
```

## Key Algorithms
- Feature extraction: MFCC (13 dims) with CMVN and mean pooling to a (13,) embedding.
- Matching: cosine similarity on L2-normalized embeddings with a user-controlled threshold.
- Multi-sample enrollment: each recording is stored; centroids are computed at load for robustness.
- Pitch tracking: YIN-based F0 estimate plotted vs time.
- AI-likelihood heuristic: combination of spectral flatness and pitch variability as a simple check.

## Data Flow
1) Enroll (Record/Import)
   - Capture audio at 16 kHz → MFCC+CMVN → 13-dim embedding
   - Store embedding in `VoiceSamples` (append) and upsert in `VoicePatterns`
   - Reload in-memory DB using centroids for each user
2) Analyze
   - Capture audio → embedding → cosine similarity vs centroids
   - Apply threshold → report best match or unknown
   - Visualize pitch curve and top-5 matches

## Database
- Tables
  - VoicePatterns(id, name, pattern)
  - VoiceSamples(id, name, pattern, created_at)
- Migrations
  - Legacy `letter` column is migrated to `name` automatically
- Helpers
  - `save_voice_pattern`, `load_database`, `list_users`, `delete_user`, `rename_user`

## UI Features
- Threshold slider (0.10–0.95) and presets (Lenient/Balanced/Strict) + auto-calibration.
- Pitch (F0) plot after capture and analysis.
- Top-5 matches chart with cosine scores.
- Import audio files (WAV/MP3/FLAC/M4A) to enroll public figures.
- User management: list, delete, rename.

## Running
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Notes on Anti-Spoofing
- Included AI-likelihood is a heuristic only. For production-grade spoofing detection, integrate a trained model (e.g., speechbrain/ASVspoof) with PyTorch.

## Roadmap
- Streaming capture with rolling pitch/VU plot
- Phrase verification (text-dependent) and calibration tools
- ECAPA-TDNN/x-vector embeddings
- Robust anti-spoofing and liveness detection
- Export/import profiles and audit logs

## License
MIT

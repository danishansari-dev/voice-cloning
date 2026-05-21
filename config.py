"""
Central configuration for the AI Voice Cloning System.

All constants, thresholds, model identifiers, and directory paths live here
so every module imports from a single source of truth.
"""

import os
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv

# ── Load environment variables from .env ────────────────────────────
load_dotenv()

HUGGINGFACE_TOKEN: str | None = os.getenv("HUGGINGFACE_TOKEN")

# ── Device selection — prefer CUDA, fall back gracefully ────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Audio sample rates ──────────────────────────────────────────────
# XTTS-v2 operates at 22050 Hz natively
SAMPLE_RATE_XTTS: int = 22050
# Bark operates at 24000 Hz natively
SAMPLE_RATE_BARK: int = 24000
# Unified rate for final stitched output
SAMPLE_RATE_UNIFIED: int = 22050

# ── Reference clip selection ────────────────────────────────────────
# XTTS needs ~6s of clean reference audio for zero-shot cloning
REFERENCE_CLIP_DURATION: float = 6.0  # seconds

# ── Audio preprocessing thresholds ──────────────────────────────────
SILENCE_THRESHOLD_DB: float = -40.0   # trim silence below this level
CHUNK_DURATION: int = 30              # seconds per analysis chunk

# ── Pause detection ─────────────────────────────────────────────────
MIN_PAUSE_DURATION_S: float = 0.15    # ignore pauses shorter than this

# ── Breathing detection ─────────────────────────────────────────────
# Fraction of mean energy below which a segment might be a breath
BREATH_ENERGY_THRESHOLD: float = 0.2

# ── Whisper post-processing ─────────────────────────────────────────
WHISPER_AMPLITUDE_FACTOR: float = 0.4  # scale amplitude for whisper effect
WHISPER_LOWPASS_HZ: int = 6000         # cutoff for whisper lowpass filter

# ── Audio stitching ─────────────────────────────────────────────────
CROSSFADE_DURATION_MS: int = 20        # overlap between consecutive speech clips
NORMALIZE_TARGET_DB: float = -3.0      # peak-normalize final output

# ── Behavioral injection probabilities ──────────────────────────────
FILLER_INJECTION_PROB: float = 0.25    # chance of injecting a filler word

# ── Common filler words used for injection ──────────────────────────
FILLER_WORDS: list[str] = [
    "uh", "um", "hmm", "like", "you know",
    "basically", "i mean",
]

# ── Model identifiers ──────────────────────────────────────────────
XTTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_ENCODER_MODEL: str = "speechbrain/spkrec-ecapa-voxceleb"
EMOTION_MODEL_PRIMARY: str = (
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
EMOTION_MODEL_FALLBACK: str = "superb/wav2vec2-base-superb-er"

# ── Directory paths (relative to project root) ─────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
PROFILES_DIR: Path = PROJECT_ROOT / "profiles"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Ensure directories exist on import so downstream code never hits FileNotFoundError
for _dir in (PROFILES_DIR, OUTPUT_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Logging configuration ───────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"
LOG_LEVEL: int = logging.INFO

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)

"""
Demo API usage for the AI Voice Cloning System.

WHY this exists:
    Provides a simple, programmatic entry point showing how to use the
    voice cloning system via Python rather than the CLI. This serves as
    a direct code integration guide for developers building applications
    on top of the system.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Configure basic logging to see pipeline progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def run_api_demo() -> None:
    """
    Demonstrates the programmatic API flow of the AI Voice Cloning System:
    1. Initialise the pipeline modules.
    2. Build a voice profile from a directory of clips.
    3. Programmatically synthesise text using the generated profile.
    4. Save the generated audio file to disk.
    """
    # ── Defer heavy ML imports so script starts up quickly ───────────
    try:
        from pipeline.profile_builder import ProfileBuilder
        from synthesis.tts_engine import TTSEngine
        import config
    except ImportError as err:
        logger.error(
            "Failed to import core modules. Make sure you are running this "
            "from the project root directory and all dependencies are installed. "
            "Error: %s",
            err,
        )
        sys.exit(1)

    speaker_name = "demo_speaker"
    reference_dir = Path("reference_clips")

    print("=" * 60)
    print("           AI VOICE CLONING SYSTEM — API DEMO")
    print("=" * 60)

    # Ensure a dummy reference directory exists or guide the user
    if not reference_dir.exists() or not any(reference_dir.iterdir()):
        reference_dir.mkdir(exist_ok=True)
        print(f"\n[!] Please place reference WAV/MP3 files of the speaker in: '{reference_dir.resolve()}'")
        print("    Then run this demo script again to build the full profile.")
        print("\nCreating a mock profile to demonstrate synthesis flow...")
        _run_mock_synthesis(speaker_name)
        return

    print(f"\n[1] Starting behavioral analysis on clips in '{reference_dir}'...")

    builder = ProfileBuilder()
    try:
        # Build behavioral + acoustic profile of the speaker
        profile = builder.build_from_directory(
            audio_dir=reference_dir,
            speaker_id=speaker_name,
            language="en",
        )

        profile_path = config.PROFILES_DIR / speaker_name / "profile.json"
        builder.save_profile(profile, profile_path)
        print(f"\n[✓] Speaker profile successfully created at: {profile_path}")

        # Summarise profile metrics
        builder.summarize(profile)

    except Exception as exc:
        logger.exception("Error during profile generation: %s", exc)
        sys.exit(1)

    print("\n[2] Initialising Synthesis Engine (XTTS-v2 + Bark)...")
    try:
        engine = TTSEngine()
    except Exception as exc:
        logger.exception("Failed to initialise TTSEngine (make sure CUDA is configured if using GPU): %s", exc)
        sys.exit(1)

    text_to_speak = (
        "Hello! I am a clone of your voice, but I also speak like you. "
        "I can take deep breaths [deep breath], pause between clauses, and "
        "even whisper when things get quiet."
    )

    print(f"\n[3] Synthesising text: '{text_to_speak}'")
    output_wav = config.OUTPUT_DIR / f"{speaker_name}_api_output.wav"

    try:
        # Synthesise speech using the speaker's behavioral profile
        audio = engine.synthesize_with_profile(
            text=text_to_speak,
            profile=profile,
            output_path=output_wav,
        )
        print(f"\n[✓] Audio successfully generated at: {output_wav.resolve()}")
        print(f"    Duration: {len(audio) / config.SAMPLE_RATE_UNIFIED:.2f}s")

    except Exception as exc:
        logger.exception("Synthesis failed: %s", exc)
        sys.exit(1)


def _run_mock_synthesis(speaker_name: str) -> None:
    """Runs a demonstration using a dummy/synthetic profile to verify code logic."""
    from pipeline.profile_builder import BehavioralProfile, ProfileBuilder
    from synthesis.tts_engine import TTSEngine
    import config

    # Construct dummy profile to verify all code paths execute correctly without real audio
    dummy_profile = BehavioralProfile(
        speaker_id=speaker_name,
        speaker_embedding=np.random.randn(192).astype(np.float32),
        reference_audio_path=str(config.PROFILES_DIR / speaker_name / "reference.wav"),
        prosody={
            "f0_mean": 180.0, "f0_std": 25.0, "f0_min": 100.0, "f0_max": 300.0, "f0_range": 200.0,
            "intensity_mean": 65.0, "intensity_std": 8.0,
            "syllables_per_second": 4.5, "words_per_minute": 140.0, "articulation_rate": 5.0,
        },
        pause_profile={
            "pause_durations": [0.2, 0.4, 0.6], "mean_pause": 0.4,
            "pause_histogram": [1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            "pre_sentence_pause_mean": 0.5, "mid_sentence_pause_mean": 0.25,
            "clause_pause_prob": 0.3, "long_pause_threshold": 0.8, "trailing_off": False
        },
        breathing_profile={
            "breath_events": [], "breath_frequency": 12.0, "pre_sentence_breath_prob": 0.6, "breath_audibility": 0.4
        },
        filler_profile={
            "filler_words": ["um", "uh"], "filler_frequency": 3.0, "filler_positions": ["sentence_start", "mid_sentence"]
        },
        emotion_fingerprint={
            "dominant_emotion": "neutral",
            "emotion_distribution": {"neutral": 0.7, "happy": 0.1, "sad": 0.1, "angry": 0.1},
            "emotional_range": 0.5, "baseline_arousal": 0.5, "baseline_valence": 0.5
        },
        whisper_threshold=0.1,
        voice_quality={"jitter": 0.01, "shimmer": 0.03, "hnr": 18.0},
        language="en",
    )

    # Save mock reference wave file so XTTS loader has something to bind to (normally copy of reference clip)
    mock_ref_path = Path(dummy_profile.reference_audio_path)
    mock_ref_path.parent.mkdir(parents=True, exist_ok=True)
    if not mock_ref_path.exists():
        # Generate 1 second of silent 22050Hz audio as placeholder reference WAV
        import soundfile as sf
        sf.write(str(mock_ref_path), np.zeros(22050, dtype=np.float32), 22050)

    print("\nInitialising TTSEngine (this will load XTTS-v2 & Bark)...")
    try:
        engine = TTSEngine()
        text = "This is a demonstration of the voice cloning synthesis pipeline."
        print(f"Synthesising mock-profile text: '{text}'")
        output_wav = config.OUTPUT_DIR / f"{speaker_name}_mock_output.wav"
        
        engine.synthesize_with_profile(
            text=text,
            profile=dummy_profile,
            output_path=output_wav,
        )
        print(f"[✓] Demo successful. Output generated at: {output_wav.resolve()}")
    except Exception as exc:
        print(f"[!] Synthesis aborted: {exc}")
        print("    (Expected if models/weights are not fully downloaded yet or CUDA is missing.)")


if __name__ == "__main__":
    run_api_demo()

"""
XTTS-v2 text-to-speech engine with behavioral profile integration.

WHY this exists:
    This is the core voice cloning engine.  It wraps Coqui's XTTS-v2
    model and adds two layers on top:

    1. Per-segment emotion conditioning — each clause gets its own
       XTTS sampling parameters (speed, temperature, etc.) derived
       from the emotion label and speaker's arousal baseline.

    2. Full pipeline orchestration — given text and a BehavioralProfile,
       it builds a synth plan (via BehavioralInjector), renders every
       segment (TTS, Bark events, pauses), stitches them together
       (via AudioStitcher), and returns the final waveform.

    The XTTS model is loaded ONCE at __init__ to avoid the multi-
    minute reload cost on each synthesis call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from scipy.signal import butter, lfilter

import config
from synthesis.text_processor import SynthSegment
from synthesis.behavioral_injector import BehavioralInjector
from synthesis.emotion_conditioner import EmotionConditioner
from synthesis.bark_events import BarkEventGenerator
from synthesis.audio_stitcher import AudioStitcher

logger = logging.getLogger(__name__)
console = Console()


class TTSEngine:
    """
    Zero-shot voice cloning engine backed by Coqui XTTS-v2.

    Handles single-segment synthesis as well as full profile-aware
    multi-segment synthesis with behavioral injection.
    """

    def __init__(self) -> None:
        """
        Load XTTS-v2 model and initialise sub-components.

        If the model fails to load the engine degrades gracefully —
        synthesize_segment returns silence and logs an error.
        """
        self._tts: Any = None
        self._model_available: bool = False
        self._sr: int = config.SAMPLE_RATE_XTTS  # 22050

        # Load the XTTS model
        try:
            with console.status("[bold cyan]Loading XTTS-v2 model…"):
                from TTS.api import TTS  # noqa: WPS433

                self._tts = TTS(config.XTTS_MODEL)
                self._tts.to(config.DEVICE)
                self._model_available = True

            logger.info("XTTS-v2 loaded on %s", config.DEVICE)
            console.print("[green]✓ XTTS-v2 model loaded successfully[/]")

        except Exception as exc:  # noqa: BLE001
            logger.error("XTTS-v2 failed to load: %s", exc)
            console.print(f"[red]✗ XTTS-v2 unavailable: {exc}[/]")

        # Sub-components for full pipeline synthesis
        self._injector = BehavioralInjector()
        self._conditioner = EmotionConditioner()
        self._bark = BarkEventGenerator()
        self._stitcher = AudioStitcher()

    # ── Public API ──────────────────────────────────────────────────

    def synthesize_segment(
        self,
        text: str,
        reference_audio_path: str | Path,
        language: str = "en",
        xtts_params: dict[str, float] | None = None,
        is_whisper: bool = False,
        output_path: str | Path | None = None,
    ) -> np.ndarray:
        """
        Synthesise a single text segment using XTTS-v2.

        @param text                 — Text to speak
        @param reference_audio_path — Path to the speaker reference wav
        @param language             — ISO language code
        @param xtts_params          — Sampling params from EmotionConditioner
        @param is_whisper           — Apply whisper post-processing
        @param output_path          — Optional path to save the segment wav
        @returns                    — Audio waveform at XTTS sample rate
        """
        if not text or not text.strip():
            logger.warning("synthesize_segment received empty text")
            return np.zeros(0, dtype=np.float32)

        if not self._model_available or self._tts is None:
            logger.error("XTTS model unavailable — returning silence")
            # Return 1 s of silence so the pipeline doesn't crash
            return np.zeros(self._sr, dtype=np.float32)

        ref_path = Path(reference_audio_path)
        if not ref_path.exists():
            logger.error("Reference audio not found: %s", ref_path)
            return np.zeros(self._sr, dtype=np.float32)

        params = xtts_params or {}

        try:
            # XTTS .tts() returns a list of floats or np.ndarray
            wav = self._tts.tts(
                text=text,
                speaker_wav=str(ref_path),
                language=language,
                speed=params.get("speed", 1.0),
                temperature=params.get("temperature", 0.65),
                top_k=int(params.get("top_k", 50)),
                top_p=params.get("top_p", 0.85),
                repetition_penalty=params.get("repetition_penalty", 2.0),
            )

            audio = np.array(wav, dtype=np.float32)

        except Exception as exc:  # noqa: BLE001
            logger.error("XTTS synthesis failed for '%s…': %s", text[:40], exc)
            return np.zeros(self._sr, dtype=np.float32)

        # Whisper post-processing — reduce amplitude, add noise, lowpass
        if is_whisper:
            audio = self._apply_whisper_effect(audio, self._sr)

        # Save to disk if requested
        if output_path is not None:
            self._stitcher.save(audio, self._sr, Path(output_path))

        logger.info(
            "Synthesised segment: %d samples (%.2f s) whisper=%s",
            len(audio),
            len(audio) / self._sr,
            is_whisper,
        )
        return audio

    def synthesize_with_profile(
        self,
        text: str,
        profile: Any,
        emotion_override: str | None = None,
        output_path: str | Path | None = None,
    ) -> np.ndarray:
        """
        Full pipeline: text + profile → behaviourally-rich waveform.

        Steps:
          1. BehavioralInjector builds a SynthPlan
          2. Each segment is rendered:
             - "tts"   → XTTS with emotion-conditioned params
             - "event" → BarkEventGenerator
             - "pause" → digital silence
          3. AudioStitcher merges everything with crossfades

        @param text              — Full text to synthesise
        @param profile           — BehavioralProfile instance
        @param emotion_override  — Force one emotion on all TTS segments
        @param output_path       — Optional path to save final wav
        @returns                 — Stitched audio at unified sample rate
        """
        if not text or not text.strip():
            logger.warning("synthesize_with_profile received empty text")
            return np.zeros(0, dtype=np.float32)

        # 1. Build the synth plan
        plan: list[SynthSegment] = self._injector.build_synth_plan(
            text, profile, emotion_override
        )

        if not plan:
            logger.warning("Empty synth plan — nothing to render")
            return np.zeros(0, dtype=np.float32)

        # Get reference audio path from the profile
        ref_path: str = getattr(profile, "reference_audio_path", "")
        if not ref_path or not Path(ref_path).exists():
            logger.error(
                "No valid reference_audio in profile ('%s'). "
                "TTS segments will be silent.",
                ref_path,
            )

        # Pre-cache common Bark events for speed
        event_cache = self._bark.cache_common_events(profile)

        # 2. Render each segment
        audio_clips: dict[int, np.ndarray] = {}

        for idx, segment in enumerate(plan):
            if segment.type == "tts":
                xtts_params = self._conditioner.get_xtts_params(
                    segment.emotion, profile
                )
                clip = self.synthesize_segment(
                    text=segment.content,
                    reference_audio_path=ref_path,
                    xtts_params=xtts_params,
                    is_whisper=segment.is_whisper,
                )
                audio_clips[idx] = clip

            elif segment.type == "event":
                # Try cache first, then generate on the fly
                if segment.content in event_cache:
                    audio_clips[idx] = event_cache[segment.content]
                else:
                    audio_clips[idx] = self._render_event(segment.content)

            elif segment.type == "pause":
                audio_clips[idx] = self._bark.generate_pause_audio(
                    segment.duration_ms
                )

            else:
                logger.warning("Unknown segment type '%s' at index %d", segment.type, idx)
                audio_clips[idx] = np.zeros(0, dtype=np.float32)

        # 3. Stitch everything together
        final = self._stitcher.stitch(
            plan, audio_clips, target_sr=config.SAMPLE_RATE_UNIFIED
        )

        # Save if requested
        if output_path is not None:
            saved = self._stitcher.save(final, config.SAMPLE_RATE_UNIFIED, Path(output_path))
            logger.info("Saved final output to %s", saved)

        logger.info(
            "synthesize_with_profile complete: %d segments → %d samples (%.2f s)",
            len(plan),
            len(final),
            len(final) / config.SAMPLE_RATE_UNIFIED if len(final) else 0,
        )
        return final

    # ── Private helpers ─────────────────────────────────────────────

    def _render_event(self, event_name: str) -> np.ndarray:
        """
        Render a Bark event by name (e.g. "breath_inhale", "filler_uh").

        Routes to the appropriate BarkEventGenerator method based
        on the event name prefix.
        """
        if event_name.startswith("breath_"):
            breath_type = event_name.replace("breath_", "")
            return self._bark.generate_breath(breath_type)

        elif event_name.startswith("filler_"):
            word = event_name.replace("filler_", "")
            return self._bark.generate_filler(word)

        elif event_name == "laugh":
            return self._bark.generate_laugh()

        else:
            logger.warning("Unknown event '%s' — returning silence", event_name)
            return np.zeros(int(self._sr * 0.5), dtype=np.float32)

    def _apply_whisper_effect(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """
        Post-process a waveform to sound whispered.

        Three transformations:
          1. Reduce amplitude to 40 % (whispers are quiet)
          2. Add subtle breathiness noise
          3. Lowpass at 6000 Hz (whispers lack high harmonics)

        @param waveform — Input audio
        @param sr       — Sample rate
        @returns        — Whisper-processed audio
        """
        if len(waveform) == 0:
            return waveform

        # 1. Reduce amplitude
        whispered = waveform * config.WHISPER_AMPLITUDE_FACTOR

        # 2. Add breathiness noise (very subtle)
        noise = np.random.normal(0, 0.005, len(whispered)).astype(np.float32)
        whispered = whispered + noise

        # 3. Lowpass filter at 6000 Hz
        try:
            nyquist = sr / 2.0
            cutoff = config.WHISPER_LOWPASS_HZ / nyquist
            # Guard against invalid cutoff (must be 0 < cutoff < 1)
            if 0 < cutoff < 1:
                b, a = butter(4, cutoff, btype="low")
                whispered = lfilter(b, a, whispered).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Lowpass filter failed: %s", exc)

        return whispered


# ── Standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from types import SimpleNamespace

    console.rule("[bold green]TTSEngine — Standalone Test")

    # NOTE: This test will only produce real audio if XTTS-v2 and Bark
    # are installed and a reference wav exists.  Without them it
    # exercises the code paths with graceful fallbacks.

    engine = TTSEngine()

    console.print(f"\n[cyan]Model available:[/] {engine._model_available}")
    console.print(f"[cyan]Sample rate:[/] {engine._sr}")

    # Test single segment (will produce silence if model isn't loaded)
    console.print("\n[yellow]Testing synthesize_segment (no model = silence)…[/]")
    dummy_ref = config.OUTPUT_DIR / "test_reference.wav"
    segment_audio = engine.synthesize_segment(
        text="Hello, this is a test.",
        reference_audio_path=dummy_ref,
        xtts_params={"speed": 1.0, "temperature": 0.65},
    )
    console.print(f"  Result: {segment_audio.shape} samples")

    # Test full pipeline with mock profile
    console.print("\n[yellow]Testing synthesize_with_profile (mock)…[/]")
    mock_fp = SimpleNamespace(dominant_emotion="happy", baseline_arousal=0.6)
    mock_profile = SimpleNamespace(
        breathing_profile={"pre_sentence_breath_prob": 0.7},
        pause_profile={"mean_pause": 0.35},
        emotion_fingerprint=mock_fp,
        filler_frequency=3.0,
        filler_words=["uh", "um"],
        trailing_off=False,
        reference_audio=str(dummy_ref),
    )

    full_audio = engine.synthesize_with_profile(
        text="This is a full pipeline test.",
        profile=mock_profile,
        output_path=config.OUTPUT_DIR / "tts_engine_test.wav",
    )
    console.print(f"  Result: {full_audio.shape} samples")

    console.rule("[bold green]Done")

"""
Bark-based paralinguistic event generator.

WHY this exists:
    Real human speech is full of non-verbal vocalisations — breaths,
    fillers ("uh", "um"), whispers, laughter.  Standard TTS models
    produce sterile, robotic output.  This module uses Suno's Bark
    model to generate those paralinguistic events as raw audio arrays
    so they can be stitched into the final waveform alongside the
    main TTS output.

    Models are loaded ONCE at construction because Bark's init is
    extremely slow (~30-60 s on GPU).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from rich.console import Console

import config

logger = logging.getLogger(__name__)
console = Console()

# Bark's native sample rate
_BARK_SR: int = config.SAMPLE_RATE_BARK  # 24000


class BarkEventGenerator:
    """
    Generates paralinguistic audio clips (breaths, fillers,
    whispers, laughs) via the Bark TTS model.

    Bark is loaded once in __init__ because model loading is
    expensive.  If Bark fails to initialise the generator
    degrades gracefully — every method returns silence and logs
    a warning.
    """

    def __init__(self) -> None:
        """
        Load Bark models into memory.

        A rich spinner is shown during loading because this step
        can take upwards of 30 seconds on a GPU machine.
        """
        self._bark_available: bool = False
        self._generate_audio: Any = None
        self._sample_rate: int = _BARK_SR

        try:
            with console.status("[bold cyan]Loading Bark models (this may take a while)…"):
                from bark import SAMPLE_RATE, generate_audio, preload_models  # noqa: WPS433

                preload_models()
                self._generate_audio = generate_audio
                self._sample_rate = SAMPLE_RATE
                self._bark_available = True

            logger.info("Bark models loaded — sample_rate=%d", self._sample_rate)
            console.print("[green]✓ Bark models loaded successfully[/]")

        except Exception as exc:  # noqa: BLE001
            # Graceful degradation — every public method returns silence
            logger.warning("Bark failed to load (%s) — all events will be silent", exc)
            console.print(f"[yellow]⚠ Bark unavailable: {exc}. Using silent fallbacks.[/]")

    # ── Public API ──────────────────────────────────────────────────

    def generate_breath(
        self,
        type: str = "inhale",
        audibility: float = 0.5,
    ) -> np.ndarray:
        """
        Generate a breathing sound.

        @param type       — "inhale", "exhale", or "sigh"
        @param audibility — 0.0-1.0 amplitude scaling
        @returns          — Audio array at Bark sample rate
        """
        prompt_map: dict[str, str] = {
            "inhale": "[deep breath]",
            "exhale": "[exhales]",
            "sigh": "[sighs]",
        }
        prompt = prompt_map.get(type, "[deep breath]")

        audio = self._safe_generate(prompt)
        # Scale by audibility so breaths can be subtle
        return (audio * np.clip(audibility, 0.0, 1.0)).astype(np.float32)

    def generate_filler(self, word: str) -> np.ndarray:
        """
        Generate a filler word vocalisation (e.g. "uh", "um").

        The throat-clear prefix nudges Bark toward a hesitant,
        natural-sounding output instead of clean speech.

        @param word — The filler word to vocalise
        @returns    — Audio array at Bark sample rate
        """
        prompt = f"[clears throat] {word}"
        return self._safe_generate(prompt)

    def generate_whisper(self, text: str) -> np.ndarray:
        """
        Generate whispered speech.

        Bark interprets markdown italics (*text*) as whispered
        delivery — this is an undocumented-but-reliable trick.

        @param text — The text to whisper
        @returns    — Audio array at Bark sample rate
        """
        prompt = f"*{text}*"
        return self._safe_generate(prompt)

    def generate_laugh(self) -> np.ndarray:
        """
        Generate a laugh clip.

        @returns — Audio array at Bark sample rate
        """
        return self._safe_generate("[laughs]")

    def generate_pause_audio(self, duration_ms: int) -> np.ndarray:
        """
        Generate pure digital silence of a specific duration.

        WHY not just concatenate zeros in the stitcher?
        This keeps the interface uniform: every segment type
        produces an ndarray at the expected sample rate.

        @param duration_ms — Silence length in milliseconds
        @returns           — Zero-filled audio array at Bark sample rate
        """
        if duration_ms <= 0:
            return np.zeros(0, dtype=np.float32)

        num_samples = int(self._sample_rate * duration_ms / 1000)
        return np.zeros(num_samples, dtype=np.float32)

    def cache_common_events(
        self,
        profile: Any,
    ) -> dict[str, np.ndarray]:
        """
        Pre-generate common events from a BehavioralProfile to avoid
        per-sentence latency during live synthesis.

        Caches: filler words from the profile, one inhale, one exhale.

        @param profile — A BehavioralProfile with filler_words list
        @returns       — Mapping of event name → pre-generated audio
        """
        cache: dict[str, np.ndarray] = {}

        # Pre-generate breathing events
        cache["breath_inhale"] = self.generate_breath("inhale")
        cache["breath_exhale"] = self.generate_breath("exhale")
        cache["breath_sigh"] = self.generate_breath("sigh")
        cache["laugh"] = self.generate_laugh()

        # Pre-generate filler words from the speaker's profile
        # filler_words is nested inside filler_profile dict
        filler_prof = getattr(profile, "filler_profile", {}) or {}
        filler_words: list[str] = filler_prof.get("filler_words", config.FILLER_WORDS)
        for word in filler_words:
            cache[f"filler_{word}"] = self.generate_filler(word)
            logger.debug("Cached filler event: %s", word)

        logger.info("Cached %d common events", len(cache))
        return cache

    # ── Private helpers ─────────────────────────────────────────────

    def _safe_generate(self, prompt: str) -> np.ndarray:
        """
        Wrap Bark generation with error handling.

        Returns one second of silence if Bark is unavailable or
        generation fails, so the pipeline never crashes.
        """
        if not self._bark_available or self._generate_audio is None:
            logger.debug("Bark unavailable — returning 1 s silence for prompt: %s", prompt)
            return np.zeros(self._sample_rate, dtype=np.float32)

        try:
            audio: np.ndarray = self._generate_audio(prompt)
            return audio.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.error("Bark generation failed for prompt '%s': %s", prompt, exc)
            return np.zeros(self._sample_rate, dtype=np.float32)


# ── Standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold green]BarkEventGenerator — Standalone Test")

    gen = BarkEventGenerator()

    console.print("\n[cyan]Generating breath (inhale)…[/]")
    breath = gen.generate_breath("inhale", audibility=0.7)
    console.print(f"  Shape: {breath.shape}  dtype: {breath.dtype}")

    console.print("[cyan]Generating filler 'uh'…[/]")
    filler = gen.generate_filler("uh")
    console.print(f"  Shape: {filler.shape}  dtype: {filler.dtype}")

    console.print("[cyan]Generating whisper…[/]")
    whisper = gen.generate_whisper("I have a secret")
    console.print(f"  Shape: {whisper.shape}  dtype: {whisper.dtype}")

    console.print("[cyan]Generating laugh…[/]")
    laugh = gen.generate_laugh()
    console.print(f"  Shape: {laugh.shape}  dtype: {laugh.dtype}")

    console.print("[cyan]Generating 500 ms pause…[/]")
    pause = gen.generate_pause_audio(500)
    console.print(f"  Shape: {pause.shape}  expected: {int(24000 * 0.5)}")

    console.rule("[bold green]Done")

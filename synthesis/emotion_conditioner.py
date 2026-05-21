"""
Emotion-to-XTTS parameter conditioning.

WHY this exists:
    XTTS-v2 has no explicit "emotion" input — instead, its generation
    quality and style are steered by sampling parameters (temperature,
    top_k, top_p, speed).  This module provides a principled mapping
    from semantic emotion labels to those low-level knobs, and then
    fine-tunes the result using the speaker's measured arousal baseline
    from their BehavioralProfile.

    Without this layer the TTS output would sound emotionally flat
    regardless of the text content.
"""

from __future__ import annotations

import logging
from typing import Any

from rich.console import Console

import config

logger = logging.getLogger(__name__)
console = Console()


# ── Emotion → XTTS parameter lookup table ───────────────────────────────
# Each entry: (speed, temperature, repetition_penalty, top_k, top_p)
_EMOTION_PARAM_TABLE: dict[str, dict[str, float]] = {
    "neutral": {
        "speed": 1.0,
        "temperature": 0.65,
        "repetition_penalty": 2.0,
        "top_k": 50,
        "top_p": 0.85,
    },
    "happy": {
        "speed": 1.15,
        "temperature": 0.8,
        "repetition_penalty": 2.0,
        "top_k": 60,
        "top_p": 0.9,
    },
    "sad": {
        "speed": 0.85,
        "temperature": 0.55,
        "repetition_penalty": 2.0,
        "top_k": 40,
        "top_p": 0.8,
    },
    "angry": {
        "speed": 1.2,
        "temperature": 0.9,
        "repetition_penalty": 2.0,
        "top_k": 70,
        "top_p": 0.95,
    },
    "whisper": {
        "speed": 0.9,
        "temperature": 0.4,
        "repetition_penalty": 2.0,
        "top_k": 30,
        "top_p": 0.75,
    },
}


class EmotionConditioner:
    """
    Maps emotion labels to XTTS-v2 sampling parameters and adjusts
    them for a specific speaker's arousal baseline.
    """

    def __init__(self) -> None:
        """No heavy init — this is a stateless mapper."""
        logger.info("EmotionConditioner initialised")

    # ── Public API ──────────────────────────────────────────────────

    def get_xtts_params(
        self,
        emotion: str,
        profile: Any,
    ) -> dict[str, float]:
        """
        Retrieve XTTS synthesis parameters for a given emotion.

        Falls back to "neutral" for unrecognised emotion labels
        so the pipeline never crashes on unexpected input.

        @param emotion — Semantic label ("happy", "sad", "angry", etc.)
        @param profile — BehavioralProfile for speaker-specific tuning
        @returns       — Dict with keys: speed, temperature,
                         repetition_penalty, top_k, top_p
        """
        base_params = _EMOTION_PARAM_TABLE.get(
            emotion.lower(),
            _EMOTION_PARAM_TABLE["neutral"],
        ).copy()

        if emotion.lower() not in _EMOTION_PARAM_TABLE:
            logger.warning(
                "Unknown emotion '%s' — falling back to neutral params",
                emotion,
            )

        # Fine-tune for the speaker's arousal baseline
        adjusted = self.adjust_for_baseline(base_params, profile)

        logger.debug(
            "XTTS params for emotion=%s: %s",
            emotion,
            adjusted,
        )
        return adjusted

    def adjust_for_baseline(
        self,
        params: dict[str, float],
        profile: Any,
    ) -> dict[str, float]:
        """
        Nudge synthesis parameters toward the speaker's baseline arousal.

        WHY: A speaker with naturally high arousal (energetic talker)
        should have slightly boosted speed and temperature even when
        delivering "neutral" content.  This prevents the cloned voice
        from sounding too subdued compared to the original.

        The adjustment is intentionally small (±10 %) to avoid
        destabilising the generation.

        @param params  — Base XTTS parameter dict (will be mutated)
        @param profile — BehavioralProfile with emotion_fingerprint
        @returns       — The same dict, adjusted in-place
        """
        # Safely extract baseline arousal (0.0-1.0 scale, 0.5 = average)
        emotion_fp = getattr(profile, "emotion_fingerprint", None)
        if emotion_fp is None:
            return params

        # emotion_fingerprint is a dict, not a namespace — use .get()
        if isinstance(emotion_fp, dict):
            baseline_arousal: float = emotion_fp.get("baseline_arousal", 0.5)
        else:
            baseline_arousal: float = getattr(emotion_fp, "baseline_arousal", 0.5)

        # Deviation from the "average" speaker — clamped to ±0.3
        deviation = max(-0.3, min(0.3, baseline_arousal - 0.5))

        # Higher arousal → slightly faster speech, warmer sampling
        # The 0.15 multiplier keeps the adjustment at ≤ ~5 %
        params["speed"] = round(params["speed"] + deviation * 0.15, 3)
        params["temperature"] = round(
            params["temperature"] + deviation * 0.1, 3
        )

        # Guard against degenerate values
        params["speed"] = max(0.5, min(2.0, params["speed"]))
        params["temperature"] = max(0.1, min(1.5, params["temperature"]))

        logger.debug(
            "Arousal adjustment: baseline=%.2f  deviation=%.2f  speed=%.3f  temp=%.3f",
            baseline_arousal,
            deviation,
            params["speed"],
            params["temperature"],
        )
        return params


# ── Standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from types import SimpleNamespace

    console.rule("[bold green]EmotionConditioner — Standalone Test")

    conditioner = EmotionConditioner()

    # Mock profile with varying arousal levels
    emotions = ["neutral", "happy", "sad", "angry", "whisper", "confused"]
    arousals = [0.3, 0.5, 0.7, 0.9]

    for arousal in arousals:
        mock_fp = SimpleNamespace(
            dominant_emotion="neutral",
            baseline_arousal=arousal,
        )
        mock_profile = SimpleNamespace(emotion_fingerprint=mock_fp)

        console.print(f"\n[bold yellow]Baseline arousal = {arousal}[/]")
        for emo in emotions:
            params = conditioner.get_xtts_params(emo, mock_profile)
            console.print(
                f"  {emo:>10s} → speed={params['speed']:.3f}  "
                f"temp={params['temperature']:.3f}  "
                f"top_k={int(params['top_k'])}  "
                f"top_p={params['top_p']:.2f}  "
                f"rep_pen={params['repetition_penalty']:.1f}"
            )

    console.rule("[bold green]Done")

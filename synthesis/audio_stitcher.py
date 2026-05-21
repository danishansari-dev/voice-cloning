"""
Audio stitcher — merges heterogeneous audio clips into one waveform.

WHY this exists:
    The synthesis pipeline produces many small audio chunks from
    different sources (XTTS at 22050 Hz, Bark at 24000 Hz, silence
    arrays).  This module handles:

      • Resampling every clip to a common target rate
      • Applying fade-out on trailing-off segments
      • Crossfading between consecutive speech segments
        for seamless transitions
      • Peak-normalising the final waveform to a broadcast-safe level
      • Saving the result as a 16-bit PCM WAV

    Without this, clip boundaries would produce audible pops,
    volume jumps, and sample-rate mismatches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from rich.console import Console
from scipy.signal import butter, lfilter

import config
from synthesis.text_processor import SynthSegment

logger = logging.getLogger(__name__)
console = Console()


class AudioStitcher:
    """
    Merges TTS + Bark event clips + pauses into one continuous
    waveform with crossfades and normalisation.
    """

    def __init__(self) -> None:
        """No heavy init — all work is per-call."""
        logger.info("AudioStitcher initialised")

    # ── Public API ──────────────────────────────────────────────────

    def stitch(
        self,
        segments: list[SynthSegment],
        audio_clips: dict[int, np.ndarray],
        target_sr: int = config.SAMPLE_RATE_UNIFIED,
    ) -> np.ndarray:
        """
        Combine ordered audio clips into a single waveform.

        Steps:
          1. Resample each clip to target_sr
          2. Apply fade_out where segment.params requests it
          3. Crossfade (20 ms) between consecutive speech segments
          4. Concatenate
          5. Normalise to –3 dB peak

        @param segments   — Ordered SynthSegment list (for metadata)
        @param audio_clips — Index → ndarray mapping from the renderer
        @param target_sr  — Output sample rate (default: unified rate)
        @returns          — Single normalised waveform
        """
        if not segments or not audio_clips:
            logger.warning("stitch() called with empty segments or clips")
            return np.zeros(0, dtype=np.float32)

        processed: list[np.ndarray] = []
        crossfade_ms = config.CROSSFADE_DURATION_MS  # 20 ms

        for idx, segment in enumerate(segments):
            clip = audio_clips.get(idx)
            if clip is None or len(clip) == 0:
                continue

            # 1. Resample to target_sr if needed
            clip = self._resample_if_needed(clip, segment, target_sr)

            # 2. Apply fade-out if the segment requests it
            if segment.params.get("fade_out", False):
                clip = self._apply_fade_out(clip, target_sr, duration_ms=200)

            # 3. Crossfade with the previous clip if both are speech
            if (
                processed
                and len(processed[-1]) > 0
                and segment.type == "tts"
                and idx > 0
                and segments[idx - 1].type == "tts"
            ):
                clip = self._crossfade(processed.pop(), clip, target_sr, crossfade_ms)

            processed.append(clip)

        if not processed:
            return np.zeros(0, dtype=np.float32)

        # 4. Concatenate
        stitched = np.concatenate(processed)

        # 5. Normalise to target dB
        stitched = self.normalize(stitched, target_db=config.NORMALIZE_TARGET_DB)

        logger.info(
            "Stitched %d clips → %d samples (%.2f s @ %d Hz)",
            len(processed),
            len(stitched),
            len(stitched) / target_sr,
            target_sr,
        )
        return stitched

    def apply_whisper_effect(
        self,
        waveform: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        """
        Make a waveform sound whispered.

        Three-step process:
          1. Reduce amplitude to 40 % — whispers are quiet
          2. Lowpass at 6000 Hz — whispers lack upper harmonics
          3. Add subtle breathiness noise — characteristic texture

        @param waveform — Input audio array
        @param sr       — Sample rate
        @returns        — Whisper-processed audio
        """
        if len(waveform) == 0:
            return waveform

        # Amplitude reduction
        whispered = waveform * config.WHISPER_AMPLITUDE_FACTOR

        # Lowpass filter
        try:
            nyquist = sr / 2.0
            cutoff = config.WHISPER_LOWPASS_HZ / nyquist
            if 0 < cutoff < 1:
                b, a = butter(4, cutoff, btype="low")
                whispered = lfilter(b, a, whispered).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Whisper lowpass failed: %s", exc)

        # Breathiness noise
        noise = np.random.normal(0, 0.005, len(whispered)).astype(np.float32)
        whispered = whispered + noise

        return whispered

    def normalize(
        self,
        waveform: np.ndarray,
        target_db: float = -3.0,
    ) -> np.ndarray:
        """
        Peak-normalise a waveform to a target dB level.

        @param waveform  — Input audio
        @param target_db — Target peak level in dBFS
        @returns         — Normalised audio
        """
        if len(waveform) == 0:
            return waveform

        peak = np.max(np.abs(waveform))
        if peak < 1e-8:
            # Essentially silence — don't amplify noise
            logger.debug("Waveform is silence, skipping normalisation")
            return waveform

        # target_db is negative; 10^(target_db/20) gives the linear gain target
        target_linear = 10.0 ** (target_db / 20.0)
        gain = target_linear / peak
        normalized = (waveform * gain).astype(np.float32)

        logger.debug(
            "Normalised: peak=%.4f → target=%.4f  gain=%.4f",
            peak,
            target_linear,
            gain,
        )
        return normalized

    def save(
        self,
        waveform: np.ndarray,
        sr: int,
        path: str | Path,
    ) -> Path:
        """
        Save waveform as a 16-bit PCM WAV file.

        @param waveform — Audio data
        @param sr       — Sample rate
        @param path     — Output file path
        @returns        — Resolved Path to the saved file
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Clip to [-1, 1] to prevent int16 overflow
        clipped = np.clip(waveform, -1.0, 1.0)

        sf.write(str(out_path), clipped, sr, subtype="PCM_16")
        logger.info("Saved WAV: %s (%d samples @ %d Hz)", out_path, len(clipped), sr)
        return out_path

    # ── Private helpers ─────────────────────────────────────────────

    def _resample_if_needed(
        self,
        clip: np.ndarray,
        segment: SynthSegment,
        target_sr: int,
    ) -> np.ndarray:
        """
        Resample a clip if its source sample rate differs from target.

        Bark events are at 24000 Hz, XTTS at 22050 Hz.
        Pause segments are rate-agnostic (silence).
        """
        if len(clip) == 0:
            return clip

        # Determine source sample rate from segment type
        if segment.type == "event":
            source_sr = config.SAMPLE_RATE_BARK  # 24000
        elif segment.type == "tts":
            source_sr = config.SAMPLE_RATE_XTTS  # 22050
        else:
            # Pauses are just zeros — no resampling needed
            return clip

        if source_sr == target_sr:
            return clip

        try:
            resampled = librosa.resample(
                clip, orig_sr=source_sr, target_sr=target_sr
            )
            return resampled.astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Resampling failed: %s — using raw clip", exc)
            return clip

    def _apply_fade_out(
        self,
        clip: np.ndarray,
        sr: int,
        duration_ms: int = 200,
    ) -> np.ndarray:
        """
        Apply a linear fade-out to the last N ms of a clip.

        Used for "trailing off" speakers whose sentences decay
        in volume toward the end.
        """
        if len(clip) == 0:
            return clip

        fade_samples = int(sr * duration_ms / 1000)
        # Don't fade more than the entire clip
        fade_samples = min(fade_samples, len(clip))

        faded = clip.copy()
        fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        faded[-fade_samples:] *= fade_curve

        return faded

    def _crossfade(
        self,
        clip_a: np.ndarray,
        clip_b: np.ndarray,
        sr: int,
        crossfade_ms: int = 20,
    ) -> np.ndarray:
        """
        Blend the tail of clip_a with the head of clip_b over
        a short crossfade window to eliminate click artefacts.

        If either clip is too short for the crossfade, they are
        simply concatenated without blending.
        """
        xfade_samples = int(sr * crossfade_ms / 1000)

        # Fall back to simple concatenation if clips are too short
        if len(clip_a) < xfade_samples or len(clip_b) < xfade_samples:
            return np.concatenate([clip_a, clip_b])

        # Linear crossfade
        fade_out = np.linspace(1.0, 0.0, xfade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, xfade_samples, dtype=np.float32)

        # Overlap region
        overlap = clip_a[-xfade_samples:] * fade_out + clip_b[:xfade_samples] * fade_in

        # Assemble: clip_a head + overlap + clip_b tail
        result = np.concatenate([
            clip_a[:-xfade_samples],
            overlap,
            clip_b[xfade_samples:],
        ])
        return result


# ── Standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    console.rule("[bold green]AudioStitcher — Standalone Test")

    stitcher = AudioStitcher()

    sr = config.SAMPLE_RATE_UNIFIED
    duration_s = 1.0
    num_samples = int(sr * duration_s)

    # Generate test signals — two 440 Hz sine waves and a pause
    t = np.linspace(0, duration_s, num_samples, dtype=np.float32)
    sine_a = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sine_b = (0.3 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
    pause = np.zeros(int(sr * 0.3), dtype=np.float32)

    segments = [
        SynthSegment(type="tts", content="Hello", emotion="neutral"),
        SynthSegment(type="pause", content="", duration_ms=300),
        SynthSegment(type="tts", content="World", emotion="happy", params={"fade_out": True}),
    ]
    clips = {0: sine_a, 1: pause, 2: sine_b}

    result = stitcher.stitch(segments, clips, target_sr=sr)
    console.print(f"[cyan]Stitched:[/] {result.shape} samples ({len(result)/sr:.2f} s)")
    console.print(f"[cyan]Peak:[/] {np.max(np.abs(result)):.4f}")

    # Test save
    out_path = config.OUTPUT_DIR / "stitcher_test.wav"
    saved = stitcher.save(result, sr, out_path)
    console.print(f"[green]Saved to:[/] {saved}")

    # Test whisper effect
    whispered = stitcher.apply_whisper_effect(sine_a, sr)
    console.print(f"[cyan]Whisper effect:[/] peak {np.max(np.abs(whispered)):.4f}")

    # Test normalisation
    quiet = sine_a * 0.01
    loud = stitcher.normalize(quiet, target_db=-3.0)
    console.print(f"[cyan]Normalise:[/] {np.max(np.abs(quiet)):.4f} → {np.max(np.abs(loud)):.4f}")

    console.rule("[bold green]Done")
